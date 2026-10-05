FROM golang:1.24.2-bookworm AS server-build
ENV CGO_ENABLED=0 GOTOOLCHAIN=local
RUN git clone --depth 1 --branch RELEASE.2025-04-22T22-12-26Z https://github.com/minio/minio.git /src
WORKDIR /src
RUN test "$(git rev-parse HEAD)" = 0d7408fc9969caf07de6a8c3a84f9fbb10a6739e \
    && go build -trimpath -ldflags "$(go run buildscripts/gen-ldflags.go 2025-04-22T22:12:26Z)" -o /out/minio .

FROM golang:1.24.2-bookworm AS client-build
ENV CGO_ENABLED=0 GOTOOLCHAIN=local
RUN git clone --depth 1 --branch RELEASE.2025-04-16T18-13-26Z https://github.com/minio/mc.git /src
WORKDIR /src
RUN test "$(git rev-parse HEAD)" = b00526b153a31b36767991a4f5ce2cced435ee8e \
    && go build -trimpath -ldflags "$(go run buildscripts/gen-ldflags.go 2025-04-16T18:13:26Z)" -o /out/mc .

FROM debian:bookworm-slim AS client
RUN apt-get update && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=client-build /out/mc /usr/local/bin/mc
COPY --from=client-build /src/LICENSE /usr/share/licenses/mc/LICENSE
ENTRYPOINT ["mc"]

FROM client AS server
COPY --from=server-build /out/minio /usr/local/bin/minio
COPY --from=server-build /src/LICENSE /usr/share/licenses/minio/LICENSE
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
