use super::{BytesFuture, Response};
use crate::{
    api_error::{ApiError, RatelimitScope, RatelimitedApiError},
    client::connector::Connector,
    error::{Error, ErrorType},
};
use http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use http_body_util::Full;
use hyper::body::Bytes;
use hyper_util::client::legacy::{Client as HyperClient, ResponseFuture as HyperResponseFuture};
use std::{
    future::{Future, Ready, ready},
    marker::PhantomData,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, ready},
    time::{Duration, Instant},
};
use tokio::time::{self, Timeout};
use twilight_http_ratelimiting::{Endpoint, Permit, PermitFuture, RateLimitHeaders, RateLimiter};

/// Parse ratelimit headers from a map of headers.
///
/// Discord only sends the scope header on 429 responses, so bucket headers
/// without a scope are parsed like a user scoped rate limit. Returns `None`
/// if the response has no bucket headers or is globally rate limited.
///
/// # Errors
///
/// Errors if a required header is missing or if a header value is of an
/// invalid type.
fn parse_ratelimit_headers(
    headers: &HeaderMap,
) -> Result<Option<RateLimitHeaders>, Box<dyn std::error::Error>> {
    match headers
        .get(RateLimitHeaders::SCOPE)
        .map(HeaderValue::as_bytes)
    {
        Some(b"global") => {
            tracing::info!("globally rate limited");

            Ok(None)
        }
        Some(b"shared") => {
            let bucket = headers
                .get(RateLimitHeaders::BUCKET)
                .ok_or("missing bucket header")?
                .as_bytes()
                .to_vec();
            let retry_after = headers
                .get(header::RETRY_AFTER)
                .ok_or("missing retry-after header")?
                .to_str()?
                .parse()?;

            Ok(Some(RateLimitHeaders::shared(bucket, retry_after)))
        }
        Some(b"user") => parse_bucket_headers(headers).map(Some),
        None if headers.contains_key(RateLimitHeaders::BUCKET) => {
            parse_bucket_headers(headers).map(Some)
        }
        _ => Ok(None),
    }
}

/// Parse the bucket, limit, remaining, and reset-after headers.
///
/// # Errors
///
/// Errors if a header is missing or if a header value is of an invalid type.
fn parse_bucket_headers(
    headers: &HeaderMap,
) -> Result<RateLimitHeaders, Box<dyn std::error::Error>> {
    let bucket = headers
        .get(RateLimitHeaders::BUCKET)
        .ok_or("missing bucket header")?
        .as_bytes()
        .to_vec();
    let limit = headers
        .get(RateLimitHeaders::LIMIT)
        .ok_or("missing limit header")?
        .to_str()?
        .parse()?;
    let remaining = headers
        .get(RateLimitHeaders::REMAINING)
        .ok_or("missing remaining header")?
        .to_str()?
        .parse()?;
    let reset_after = headers
        .get(RateLimitHeaders::RESET_AFTER)
        .ok_or("missing reset-after header")?
        .to_str()?
        .parse()?;

    Ok(RateLimitHeaders {
        bucket,
        limit,
        remaining,
        reset_at: Instant::now() + Duration::from_secs_f32(reset_after),
    })
}

/// Parse the error body of a response with the given status.
///
/// A 429 body may also carry an error code, which the untagged [`ApiError`]
/// would parse as [`ApiError::General`], so it is parsed as
/// [`RatelimitedApiError`] first. `scope` comes from the response headers.
fn parse_api_error(
    status: StatusCode,
    scope: Option<RatelimitScope>,
    body: &[u8],
) -> Result<ApiError, crate::json::JsonError> {
    if status == StatusCode::TOO_MANY_REQUESTS
        && let Ok(mut error) = crate::json::from_bytes::<RatelimitedApiError>(body)
    {
        error.scope = scope;
        return Ok(ApiError::Ratelimited(error));
    }

    crate::json::from_bytes(body)
}

/// Sub-futures of [`ResponseFuture`].
enum ResponseStageFuture {
    /// Future that completes with an error response body.
    Error {
        /// Inner response body future.
        fut: BytesFuture,
        /// Erroneous response status code.
        status: StatusCode,
        /// Rate limit scope from the response headers.
        scope: Option<RatelimitScope>,
    },
    /// Future that completes when a rate limit permit is ready.
    RateLimitPermit(PermitFuture),
    /// Future that completes with a response or timeout.
    Response {
        /// Inner timed response future.
        fut: Pin<Box<Timeout<HyperResponseFuture>>>,
        /// Optional rate limit permit.
        permit: Option<Permit>,
    },
}

/// [`PermitFuture`] generator.
struct PermitFutureGenerator {
    /// Rate limiter to acquire permits from.
    rate_limiter: RateLimiter,
    /// Rate limiter endpoint to acquire permits for.
    endpoint: Endpoint,
}

impl PermitFutureGenerator {
    /// Generates a permit future.
    fn generate(&self) -> PermitFuture {
        tracing::debug!("awaiting permit");
        self.rate_limiter.acquire(self.endpoint.clone())
    }
}

/// [`Timeout<HyperResponseFuture>`] generator.
struct TimedResponseFutureGenerator {
    /// HTTP client to send requests from.
    client: HyperClient<Connector, Full<Bytes>>,
    /// HTTP request to send.
    request: Request<Full<Bytes>>,
    /// Duration after which the request times out.
    timeout: Duration,
}

impl TimedResponseFutureGenerator {
    /// Generates a timeout response future.
    fn generate(&self) -> Pin<Box<Timeout<HyperResponseFuture>>> {
        tracing::debug!("awaiting response");
        Box::pin(time::timeout(
            self.timeout,
            self.client.request(self.request.clone()),
        ))
    }
}

/// Future that completes when a [`Response`] is received.
///
/// # Rate limits
///
/// Requests that exceed a rate limit are automatically and immediately retried
/// until they succeed or fail with another error. If configured without a
/// [`RateLimiter`], care must be taken that an external service intercepts and
/// delays these retry requests. Disable retrying via
/// [`ClientBuilder::retry_rate_limited`] to instead return these responses as
/// [`ErrorType::Response`] errors.
///
/// # Canceling a response future pre-flight
///
/// Response futures can be canceled pre-flight via
/// [`ResponseFuture::set_pre_flight`]. This allows you to cancel requests that
/// are no longer necessary once they have been cleared by the ratelimit queue,
/// which may be necessary in scenarios where requests are being spammed. Refer
/// to its documentation for more information.
///
/// # Errors
///
/// Returns an [`ErrorType::Parsing`] error type if the request failed and the
/// error in the response body could not be deserialized.
///
/// Returns an [`ErrorType::RequestCanceled`] error type if the request was
/// canceled by the user.
///
/// Returns an [`ErrorType::RequestError`] error type if creating the request
/// failed.
///
/// Returns an [`ErrorType::RequestTimedOut`] error type if the request timed
/// out. The timeout value is configured via [`ClientBuilder::timeout`].
///
/// Returns an [`ErrorType::Response`] error type if the request failed.
///
/// [`ClientBuilder::retry_rate_limited`]: crate::client::ClientBuilder::retry_rate_limited
/// [`ClientBuilder::timeout`]: crate::client::ClientBuilder::timeout
/// [`ErrorType::Json`]: crate::error::ErrorType::Json
/// [`ErrorType::Parsing`]: crate::error::ErrorType::Parsing
/// [`ErrorType::RequestCanceled`]: crate::error::ErrorType::RequestCanceled
/// [`ErrorType::RequestError`]: crate::error::ErrorType::RequestError
/// [`ErrorType::RequestTimedOut`]: crate::error::ErrorType::RequestTimedOut
/// [`ErrorType::Response`]: crate::error::ErrorType::Response
/// [`Response`]: super::Response
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct ResponseFuture<T>(Result<Inner<T>, Ready<Error>>);

impl<T> ResponseFuture<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: HyperClient<Connector, Full<Bytes>>,
        invalid_token: Option<Arc<AtomicBool>>,
        request: Request<Full<Bytes>>,
        span: tracing::Span,
        timeout: Duration,
        rate_limiter: Option<RateLimiter>,
        retry_rate_limited: bool,
        endpoint: Endpoint,
    ) -> Self {
        let entered = span.entered();

        let permit_generator = rate_limiter.map(|rate_limiter| PermitFutureGenerator {
            rate_limiter,
            endpoint,
        });
        let response_generator = TimedResponseFutureGenerator {
            client,
            request,
            timeout,
        };
        let stage = permit_generator.as_ref().map_or_else(
            || ResponseStageFuture::Response {
                fut: response_generator.generate(),
                permit: None,
            },
            |generator| ResponseStageFuture::RateLimitPermit(generator.generate()),
        );
        Self(Ok(Inner {
            invalid_token,
            permit_generator,
            phantom: PhantomData,
            pre_flight_check: None,
            response_generator,
            retry_rate_limited,
            span: entered.exit(),
            stage,
        }))
    }

    /// Set a function to call after clearing the ratelimiter but prior to
    /// sending the request to determine if the request is still valid.
    ///
    /// This function will be a no-op if the request has failed, has already
    /// passed the ratelimiter, or if there is no ratelimiter configured.
    ///
    /// Returns whether the pre flight function was set.
    ///
    /// # Examples
    ///
    /// Delete a message, but immediately before sending the request check if
    /// the request should still be sent:
    ///
    /// ```no_run
    /// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::{
    ///     collections::HashSet,
    ///     env,
    ///     future::IntoFuture,
    ///     sync::{Arc, Mutex},
    /// };
    /// use twilight_http::{Client, error::ErrorType};
    /// use twilight_model::id::Id;
    ///
    /// let channel_id = Id::new(1);
    /// let message_id = Id::new(2);
    ///
    /// let channels_ignored = {
    ///     let mut map = HashSet::new();
    ///     map.insert(channel_id);
    ///
    ///     Arc::new(Mutex::new(map))
    /// };
    ///
    /// let client = Client::new(env::var("DISCORD_TOKEN")?);
    /// let mut req = client.delete_message(channel_id, message_id).into_future();
    ///
    /// let channels_ignored_clone = channels_ignored.clone();
    /// req.set_pre_flight(move || {
    ///     // imagine you have some logic here to external state that checks
    ///     // whether the request should still be performed
    ///     let channels_ignored = channels_ignored_clone.lock().expect("channels poisoned");
    ///
    ///     !channels_ignored.contains(&channel_id)
    /// });
    ///
    /// // the pre-flight check will cancel the request
    /// assert!(matches!(
    ///     req.await.unwrap_err().kind(),
    ///     ErrorType::RequestCanceled,
    /// ));
    /// # Ok(()) }
    /// ```
    pub fn set_pre_flight<P>(&mut self, predicate: P) -> bool
    where
        P: Fn() -> bool + Send + 'static,
    {
        if let Ok(inner) = &mut self.0
            && inner.permit_generator.is_some()
            && inner.pre_flight_check.is_none()
        {
            inner.pre_flight_check = Some(Box::new(predicate));

            true
        } else {
            false
        }
    }

    /// Creates a future that is immediately ready with an error.
    pub(crate) fn error(source: Error) -> Self {
        Self(Err(ready(source)))
    }
}

impl<T: Unpin> Future for ResponseFuture<T> {
    type Output = Result<Response<T>, Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let inner = match &mut self.0 {
            Ok(inner) => inner,
            Err(err) => return Pin::new(err).poll(cx).map(Err),
        };

        let _entered = inner.span.enter();

        loop {
            match &mut inner.stage {
                ResponseStageFuture::Error { fut, status, scope } => {
                    let body = ready!(Pin::new(fut).poll(cx)).map_err(|source| Error {
                        kind: ErrorType::RequestError,
                        source: Some(Box::new(source)),
                    })?;

                    return Poll::Ready(Err(match parse_api_error(*status, *scope, &body) {
                        Ok(error) => Error {
                            kind: ErrorType::Response {
                                body,
                                error,
                                status: super::StatusCode::new(status.as_u16()),
                            },
                            source: None,
                        },
                        Err(source) => Error {
                            kind: ErrorType::Parsing { body },
                            source: Some(Box::new(source)),
                        },
                    }));
                }
                ResponseStageFuture::RateLimitPermit(fut) => {
                    let permit = ready!(Pin::new(fut).poll(cx));
                    if inner
                        .pre_flight_check
                        .as_ref()
                        .is_some_and(|check| !check())
                    {
                        return Poll::Ready(Err(Error {
                            kind: ErrorType::RequestCanceled,
                            source: None,
                        }));
                    }

                    inner.stage = ResponseStageFuture::Response {
                        fut: inner.response_generator.generate(),
                        permit: Some(permit),
                    };
                }
                ResponseStageFuture::Response { fut, permit } => {
                    let response = ready!(Pin::new(fut).poll(cx))
                        .map_err(|source| Error {
                            kind: ErrorType::RequestTimedOut,
                            source: Some(Box::new(source)),
                        })?
                        .map_err(|source| Error {
                            kind: ErrorType::RequestError,
                            source: Some(Box::new(source)),
                        })?;

                    if response.status() == StatusCode::UNAUTHORIZED
                        && let Some(invalid) = &inner.invalid_token
                    {
                        invalid.store(true, Ordering::Relaxed);
                    }

                    if let Some(permit) = permit.take() {
                        match parse_ratelimit_headers(response.headers()) {
                            Ok(v) => permit.complete(v),
                            Err(source) => {
                                tracing::warn!("header parsing failed: {source}; {response:?}");

                                permit.complete(None);
                            }
                        }
                    }

                    if response.status().is_success() {
                        #[cfg(feature = "decompression")]
                        let mut response = response;
                        // Inaccurate since end-users can only access the decompressed body.
                        #[cfg(feature = "decompression")]
                        response.headers_mut().remove(header::CONTENT_LENGTH);

                        return Poll::Ready(Ok(Response::new(response)));
                    } else if response.status() == StatusCode::TOO_MANY_REQUESTS
                        && inner.retry_rate_limited
                    {
                        tracing::info!("rate limited; retrying");
                        inner.stage = match &inner.permit_generator {
                            Some(generator) => {
                                ResponseStageFuture::RateLimitPermit(generator.generate())
                            }
                            None => ResponseStageFuture::Response {
                                fut: inner.response_generator.generate(),
                                permit: None,
                            },
                        };
                    } else {
                        inner.stage = ResponseStageFuture::Error {
                            status: response.status(),
                            scope: response
                                .headers()
                                .get(RateLimitHeaders::SCOPE)
                                .and_then(|value| RatelimitScope::from_header(value.as_bytes())),
                            fut: Response::<()>::new(response).bytes(),
                        };
                    }
                }
            }
        }
    }
}

/// Internal response future fields.
struct Inner<T> {
    /// Whether the client's token is invalidated.
    invalid_token: Option<Arc<AtomicBool>>,
    /// Optional [`PermitFuture`] generator, if registered.
    permit_generator: Option<PermitFutureGenerator>,
    phantom: PhantomData<T>,
    /// Predicate to check after completing [`ResponseStageFuture::RateLimitPermit`].
    pre_flight_check: Option<Box<dyn Fn() -> bool + Send + 'static>>,
    /// [`Timeout<HyperResponseFuture>`] generator.
    response_generator: TimedResponseFutureGenerator,
    /// Whether to retry requests that receive a 429 response.
    retry_rate_limited: bool,
    /// This future's span.
    span: tracing::Span,
    /// This future's current stage.
    stage: ResponseStageFuture,
}

#[cfg(test)]
mod tests {
    use super::parse_ratelimit_headers;
    use crate::{
        Client,
        api_error::{ApiError, RatelimitScope, RatelimitedApiError},
        error::ErrorType,
    };
    use http::{HeaderMap, HeaderValue, header::RETRY_AFTER};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };
    use twilight_http_ratelimiting::RateLimitHeaders;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|&(name, value)| (name.parse().unwrap(), HeaderValue::from_static(value)))
            .collect()
    }

    const BUCKET_HEADERS: [(&str, &str); 4] = [
        (RateLimitHeaders::BUCKET, "abcd1234"),
        (RateLimitHeaders::LIMIT, "5"),
        (RateLimitHeaders::REMAINING, "4"),
        (RateLimitHeaders::RESET_AFTER, "1.5"),
    ];

    #[test]
    fn bucket_headers_without_scope() {
        let parsed = parse_ratelimit_headers(&headers(&BUCKET_HEADERS))
            .unwrap()
            .unwrap();

        assert_eq!(parsed.bucket, b"abcd1234");
        assert_eq!(parsed.limit, 5);
        assert_eq!(parsed.remaining, 4);
    }

    #[test]
    fn no_headers() {
        assert!(
            parse_ratelimit_headers(&HeaderMap::new())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn user_scope() {
        let mut map = headers(&BUCKET_HEADERS);
        map.insert(RateLimitHeaders::SCOPE, HeaderValue::from_static("user"));
        let parsed = parse_ratelimit_headers(&map).unwrap().unwrap();

        assert_eq!(parsed.bucket, b"abcd1234");
        assert_eq!(parsed.remaining, 4);
    }

    #[test]
    fn user_scope_missing_bucket_headers() {
        let map = headers(&[(RateLimitHeaders::SCOPE, "user")]);

        assert!(parse_ratelimit_headers(&map).is_err());
    }

    #[test]
    fn shared_scope() {
        let map = headers(&[
            (RateLimitHeaders::SCOPE, "shared"),
            (RateLimitHeaders::BUCKET, "abcd1234"),
            (RETRY_AFTER.as_str(), "3"),
        ]);
        let parsed = parse_ratelimit_headers(&map).unwrap().unwrap();

        assert_eq!(parsed.bucket, b"abcd1234");
        assert_eq!(parsed.limit, 0);
        assert_eq!(parsed.remaining, 0);
    }

    #[test]
    fn global_scope() {
        let mut map = headers(&BUCKET_HEADERS);
        map.insert(RateLimitHeaders::SCOPE, HeaderValue::from_static("global"));

        assert!(parse_ratelimit_headers(&map).unwrap().is_none());
    }

    /// Sends one request to a server that answers 429 with `scope` and `body`.
    async fn rate_limited_without_retry(scope: &'static str, body: &'static str) -> crate::Error {
        let response = format!(
            "HTTP/1.1 429 Too Many Requests\r\n\
            content-type: application/json\r\n\
            content-length: {}\r\n\
            retry-after: 1\r\n\
            x-ratelimit-scope: {scope}\r\n\r\n{body}",
            body.len(),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0; 1024];
            let _ = stream.read(&mut buf).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let client = Client::builder()
            .proxy(addr.to_string(), true)
            .retry_rate_limited(false)
            .token("Bot foo".to_owned())
            .build();
        let error = client.current_user().await.unwrap_err();
        server.await.unwrap();

        error
    }

    fn assert_rate_limited(error: &crate::Error, expected: Option<RatelimitScope>) {
        assert!(matches!(
            error.kind(),
            ErrorType::Response {
                error: ApiError::Ratelimited(RatelimitedApiError { retry_after, scope, .. }),
                status,
                ..
            } if (retry_after - 1.0).abs() < f64::EPSILON
                && status.get() == 429
                && *scope == expected,
        ));
    }

    #[tokio::test]
    async fn rate_limited() {
        let error = rate_limited_without_retry(
            "user",
            r#"{"global":false,"message":"You are being rate limited.","retry_after":1}"#,
        )
        .await;

        assert_rate_limited(&error, Some(RatelimitScope::User));
    }

    #[tokio::test]
    async fn rate_limited_with_code() {
        let error = rate_limited_without_retry(
            "shared",
            r#"{"code":20028,"global":false,"message":"The write action you are performing on the channel has hit the write rate limit.","retry_after":1}"#,
        )
        .await;

        assert_rate_limited(&error, Some(RatelimitScope::Shared));
    }

    #[tokio::test]
    async fn rate_limited_with_unknown_scope() {
        let error = rate_limited_without_retry(
            "new-scope",
            r#"{"global":false,"message":"You are being rate limited.","retry_after":1}"#,
        )
        .await;

        assert_rate_limited(&error, None);
    }
}
