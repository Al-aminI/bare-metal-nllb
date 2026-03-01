# Makefile -- PicoNLLB
#
# Targets:
#   make              build for host (macOS / Linux)
#   make optimized    build with all optimizations (threading + NEON on ARM)
#   make arm          cross-compile for ARM (Raspberry Pi)
#   make debug        build with sanitizers
#   make clean        remove artifacts

CC      = cc
TARGET  = pico_nllb
SRCS    = main.c loader.c tensor.c encoder.c decoder.c
OBJS    = $(SRCS:.c=.o)

# ─── Host ──────────────────────────────────────────────────────────────────
CFLAGS  = -O2 -std=c11 -Wall -Wextra -ffast-math -DNDEBUG
LDFLAGS = -lm

# ─── Optimized (threading + NEON on ARM) ──────────────────────────────────
OPT_CFLAGS = -O3 -std=c11 -Wall -Wextra -ffast-math -DNDEBUG -pthread
OPT_LDFLAGS = -lm -lpthread

# Detect ARM and add NEON flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    OPT_CFLAGS += -march=armv8-a+simd
endif
ifeq ($(UNAME_M),aarch64)
    OPT_CFLAGS += -march=armv8-a+simd
endif

# ─── ARM cross-compile ────────────────────────────────────────────────────
ARM_CC     = aarch64-linux-gnu-gcc
ARM_TARGET = pico_nllb_arm
ARM_CFLAGS = -O3 -march=armv8-a+simd -std=c11 -Wall -ffast-math -DNDEBUG -pthread
ARM_LDFLAGS = -lm -lpthread

# ─── Debug ────────────────────────────────────────────────────────────────
DEBUG_CFLAGS = -O0 -g3 -std=c11 -Wall -Wextra \
               -fsanitize=address,undefined -DDEBUG

.PHONY: all optimized fp16 arm debug clean

all: $(TARGET)
	@echo ""
	@echo "Built: ./$(TARGET)"
	@echo "Usage: ./$(TARGET) model.safetensors eng_Latn fra_Latn 256047 59002 4 2"

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c pico.h
	$(CC) $(CFLAGS) -c -o $@ $<

optimized: $(SRCS) pico.h
	$(CC) $(OPT_CFLAGS) -o $(TARGET)_opt $(SRCS) $(OPT_LDFLAGS)
	@echo ""
	@echo "Built optimized: ./$(TARGET)_opt"
	@echo "Optimizations: Fused dequant+dot, Multi-threading (4 cores)"
	@if [ "$(UNAME_M)" = "arm64" ] || [ "$(UNAME_M)" = "aarch64" ]; then \
		echo "               NEON SIMD (ARM)"; \
	fi
	@echo ""
	@echo "Usage: ./$(TARGET)_opt model.safetensors eng_Latn fra_Latn 256047 59002 4 2"

fp16: main_fp16.c loader.c tensor.c encoder.c decoder.c decoder_fp16.c pico.h fp16.h
	$(CC) $(OPT_CFLAGS) -o $(TARGET)_fp16 main_fp16.c loader.c tensor.c encoder.c decoder.c decoder_fp16.c $(OPT_LDFLAGS)
	@echo ""
	@echo "Built FP16: ./$(TARGET)_fp16"
	@echo "Optimizations: All Phase 1 + FP16 KV cache (50% memory reduction)"
	@echo "Memory: 130MB → 82MB (37% reduction)"
	@if [ "$(UNAME_M)" = "arm64" ] || [ "$(UNAME_M)" = "aarch64" ]; then \
		echo "SIMD: NEON enabled"; \
	fi
	@echo ""
	@echo "Usage: ./$(TARGET)_fp16 model.safetensors eng_Latn fra_Latn 256047 59002 4 2"

arm: $(SRCS) pico.h
	$(ARM_CC) $(ARM_CFLAGS) -o $(ARM_TARGET) $(SRCS) $(ARM_LDFLAGS)
	@echo "Built ARM: ./$(ARM_TARGET)"
	@echo "Optimizations: Fused dequant+dot, Multi-threading, NEON SIMD"

debug: $(SRCS) pico.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET)_debug $(SRCS) $(LDFLAGS)
	@echo "Built debug: ./$(TARGET)_debug"

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET)_opt $(TARGET)_fp16 $(TARGET)_debug $(ARM_TARGET)
