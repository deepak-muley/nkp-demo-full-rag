.PHONY: build push package chart-push release

VERSION ?= 1.0.0
IMAGE ?= ghcr.io/deepak-muley/demo-full-rag
CHART_REGISTRY ?= oci://ghcr.io/deepak-muley/charts

build:
	docker build -t $(IMAGE):$(VERSION) .
	docker tag $(IMAGE):$(VERSION) $(IMAGE):latest

push: build
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest

package:
	helm package chart/ --version $(VERSION)

chart-push: package
	helm push demo-full-rag-$(VERSION).tgz $(CHART_REGISTRY)
	rm -f demo-full-rag-$(VERSION).tgz

release: push chart-push
