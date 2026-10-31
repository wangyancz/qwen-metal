CC := clang
AR := ar
PYTHON ?= python3
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm -lz

# Vendored from h3-metal (MIT, Salvatore Sanfilippo): the Metal runtime,
# safetensors/weight plumbing, media pipeline, terminal preview, tokenizer,
# and the linenoise REPL editor. Kept under their original h3_ names.
VENDORED_C := h3_safetensors.c h3_weights.c h3_ffmpeg.c h3_terminal.c \
	linenoise.c
VENDORED_M := h3_gpu.m h3_metal.m h3_tokenizer.m
QWEN_C := qwen_host.c qwen_json.c qwen_model.c qwen_png.c qwen_vision.c qwen_imgproc.c
QWEN_M := qwen_dit.m qwen_text_encoder.m qwen_vae.m \
	qwen_pipeline.m
LIB_OBJ := $(VENDORED_C:.c=.o) $(VENDORED_M:.m=.o) $(QWEN_C:.c=.o) $(QWEN_M:.m=.o)
CLI_OBJ := main.o

.PHONY: all host-tests dit-fixture dit-tests text-fixture text-tests \
	vlm-fixture vlm-tests vae-fixture vae-tests pipeline-tests \
	vision-fixture vision-tests lora-tests clean

all: qwen libqwen.a

qwen: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libqwen.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

tests/test_host: tests/test_host.o qwen_host.o
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_dit_block: tests/test_dit_block.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_text_encoder: tests/test_text_encoder.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_text_vlm: tests/test_text_vlm.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_imgproc: tests/test_imgproc.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_ti2i_text: tests/test_ti2i_text.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_prefix_kernel: tests/test_prefix_kernel.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_vae: tests/test_vae.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_ti2i_fixture: tests/test_ti2i_fixture.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

ti2ifixture-tests: tests/test_ti2i_fixture
	./tests/test_ti2i_fixture

tests/test_lora: tests/test_lora.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

lora-tests: tests/test_lora
	./tests/test_lora

tests/test_png: tests/test_png.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

png-tests: tests/test_png
	./tests/test_png

host-tests: tests/test_host
	./tests/test_host

dit-fixture:
	$(PYTHON) tests/gen_dit_fixture.py

dit-tests: tests/test_dit_block
	./tests/test_dit_block

text-fixture:
	$(PYTHON) tests/gen_text_fixture.py

vae-fixture:
	$(PYTHON) tests/gen_vae_fixture.py

text-tests: tests/test_text_encoder
	./tests/test_text_encoder

vlm-fixture:
	$(PYTHON) tests/gen_text_vlm_fixture.py

vlm-tests: tests/test_text_vlm
	./tests/test_text_vlm

imgproc-fixture:
	$(PYTHON) tests/gen_imgproc_fixture.py

imgproc-tests: tests/test_imgproc
	./tests/test_imgproc

ti2itext-tests: tests/test_ti2i_text
	./tests/test_ti2i_text

prefix-kernel-test: tests/test_prefix_kernel
	./tests/test_prefix_kernel

vae-tests: tests/test_vae
	./tests/test_vae

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

-include $(wildcard *.d tests/*.d)

clean:
	rm -f libqwen.a *.o *.d tests/*.o tests/*.d tests/test_host

PIPELINE_TEST_SRC := tests/test_pipeline.c
BENCH_LINEAR_SRC := tests/bench_linear.c
PIPELINE_FIXTURE := tests/pipeline_fixture/rope_cos_4.f32
PIPELINE_TEST_OBJ := $(PIPELINE_TEST_SRC:.c=.o)

tests/test_pipeline: $(PIPELINE_TEST_OBJ) libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

pipeline-tests: tests/test_pipeline
	./tests/test_pipeline

tests/test_vision: tests/test_vision.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

vision-tests: tests/test_vision
	./tests/test_vision
tests/bench_linear: tests/bench_linear.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

bench-linear: tests/bench_linear
	./tests/bench_linear

tests/bench_sdpa: tests/bench_sdpa.o libqwen.a
	$(CC) -o $@ $^ $(LDLIBS)

bench-sdpa: tests/bench_sdpa
	./tests/bench_sdpa
