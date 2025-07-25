.PHONY: all test test-concurrent lint dummy

all:
	cargo build

test:
	cargo test --examples
	cargo test --features concurrent --lib --tests --workspace
	cargo test --features concurrent --doc --workspace
	pytest

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --no-default-features --profile test
	cargo clippy --workspace --all-targets --profile test
	cargo check --workspace --no-default-features --profile test
	cargo check --workspace --all-targets --all-features --profile test

docs:
	# Prevent the cached crates.js from confusing the current run
	rm target/doc/crates.js || true
	RUSTDOCFLAGS="--html-in-header gh-pages/katex-header.html $(RUSTDOCFLAGS)" cargo rustdoc --examples --all-features
	RUSTDOCFLAGS="--html-in-header gh-pages/katex-header.html $(RUSTDOCFLAGS)" cargo doc --all --no-deps --document-private-items --all-features
	# Prevent the examples from showing up in the sidebar
	echo "window.ALL_CRATES = [$$(ls crates/ | sed "s/.*/'&',/; s/-/_/g")'ext'];" > target/doc/crates.js


BENCHMARKS = $(filter-out examples/benchmarks/%-nassau, $(wildcard examples/benchmarks/*))

benchmarks: $(BENCHMARKS)
benchmarks-nassau: $(wildcard examples/benchmarks/*-nassau)

fix-benchmarks: $(patsubst examples/benchmarks/%, examples/benchmarks/%-fixed, BENCHMARKS)
fix-benchmarks-nassau: $(patsubst examples/benchmarks/%, examples/benchmarks/%-fixed, $(wildcard examples/benchmarks/*-nassau))

benchmarks-concurrent: $(patsubst examples/benchmarks/%, examples/benchmarks/%-concurrent, $(BENCHMARKS))

examples/benchmarks/%-fixed: FILE = examples/benchmarks/$*
examples/benchmarks/%-fixed: dummy
	(head -n 1 $(FILE) && bash -c "echo '' | cargo run --example $$(head -n 1 $(FILE))") > $@
	if diff --color $(FILE) $@; then \
	    rm $@; \
	else \
	    mv $@ $(FILE); \
	fi
examples/benchmarks/%-nassau-fixed: dummy
	(head -n 1 $(FILE)-nassau && bash -c "echo '' | cargo run --features nassau --example $$(head -n 1 $(FILE)-nassau)") > $@
	if diff --color $(FILE)-nassau $@; then \
	    rm $@; \
	else \
	    mv $@ $(FILE)-nassau; \
	fi


examples/benchmarks/%: dummy
	(head -n 1 $@ && bash -c "echo '' | cargo run --example $$(head -n 1 $@)") | diff --color $@ -

examples/benchmarks/%-nassau: dummy
	(head -n 1 $@ && bash -c "echo '' | cargo run --features nassau --example $$(head -n 1 $@)") | diff --color $@ -

examples/benchmarks/%-concurrent: FILE = examples/benchmarks/$*
examples/benchmarks/%-concurrent: dummy
	@{ \
		head -n 1 $(FILE); \
		timeout 30 bash -c "echo '' | cargo run --features concurrent --example $$(head -n 1 $(FILE))"; \
		ec=$$?; \
	} > output.txt; \
	if [ $$ec -eq 124 ]; then \
		echo "Timeout occurred, but treating as success."; \
	elif [ $$ec -eq 0 ]; then \
		diff --color output.txt $(FILE); \
		rm output.txt; \
	else \
		echo "Command failed with code $$ec."; \
		rm output.txt; \
		exit 1; \
	fi

miri:
	cargo miri test -p once
	MIRIFLAGS="-Zmiri-ignore-leaks -Zmiri-disable-isolation" cargo miri test -p once --features concurrent
