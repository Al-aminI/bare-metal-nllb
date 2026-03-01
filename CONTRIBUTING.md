# Contributing to MetalNLLB

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/MetalNLLB.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Development Setup

```bash
# Install dependencies (macOS)
brew install make

# Build
make optimized

# Run tests
python benchmarks/test_hausa_translation.py
```

## Code Style

- **C Code**: Follow K&R style with 4-space indentation
- **Comments**: Use `/* */` for multi-line, `//` for single-line
- **Naming**: `snake_case` for functions/variables, `UPPER_CASE` for constants
- **Line Length**: Max 100 characters

## Testing Requirements

All contributions must:

1. **Maintain Quality**: Pass all quality tests (100% parity with CTranslate2)
2. **Include Tests**: Add tests for new features
3. **Document Changes**: Update README and relevant docs
4. **Pass Benchmarks**: No performance regressions

### Running Tests

```bash
# Quality tests (must pass 5/5)
python benchmarks/test_hausa_translation.py

# Performance benchmarks
python benchmarks/benchmark_direct.py
python benchmarks/benchmark_performance.py

# Build all variants
make all
```

## Areas for Contribution

### High Priority

1. **ARM/NEON Testing**
   - Test on Raspberry Pi 4/5
   - Validate NEON SIMD performance
   - Report actual speedup numbers

2. **GPU Acceleration**
   - CUDA kernels for matmul
   - Metal shaders for macOS/iOS
   - OpenCL for portability

3. **Quantization**
   - INT4 weights (with quality validation)
   - Mixed precision (FP16 self-attn, FP32 cross-attn)
   - Dynamic quantization

### Medium Priority

4. **Tokenizer Integration**
   - SentencePiece in C
   - Eliminate Python dependency
   - End-to-end C pipeline

5. **Model Support**
   - NLLB-1.3B, NLLB-3.3B
   - Other encoder-decoder models (mBART, M2M100)
   - Decoder-only models (for comparison)

6. **Features**
   - Dynamic beam size
   - Sampling methods (temperature, top-p, top-k)
   - Batch processing

### Low Priority

7. **Deployment**
   - Mobile builds (iOS/Android)
   - WebAssembly port
   - Microcontroller support (ESP32, STM32)

8. **Documentation**
   - API documentation
   - Tutorial notebooks
   - Video walkthroughs

## Pull Request Process

1. **Update Documentation**: README, code comments, research report if applicable
2. **Add Tests**: Ensure new features are tested
3. **Run Benchmarks**: Verify no performance regression
4. **Quality Check**: All tests must pass (5/5 exact matches)
5. **Describe Changes**: Clear PR description with motivation and impact

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Include file names

## Testing
- [ ] Quality tests pass (5/5)
- [ ] Performance benchmarks run
- [ ] No regressions

## Performance Impact
- Baseline: X tok/s
- After changes: Y tok/s
- Speedup: Z%

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Benchmarks run
```

## Bug Reports

When reporting bugs, include:

1. **Environment**: OS, CPU, compiler version
2. **Steps to Reproduce**: Exact commands to trigger bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Logs**: Full error messages and stack traces

## Feature Requests

When requesting features, include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Impact**: Performance/memory/complexity trade-offs

## Code Review Process

All PRs will be reviewed for:

1. **Correctness**: Does it work as intended?
2. **Quality**: Does it maintain 100% parity?
3. **Performance**: Does it improve or maintain speed?
4. **Code Quality**: Is it readable and maintainable?
5. **Documentation**: Is it well-documented?

## Performance Guidelines

- **No Regressions**: New code must not slow down existing functionality
- **Benchmark Everything**: Measure before and after
- **Profile First**: Use profiling tools to identify bottlenecks
- **Optimize Carefully**: Maintain code readability

## Questions?

- Open an issue for questions
- Tag with `question` label
- Be specific and provide context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Acknowledged in research report updates

Thank you for contributing to MetalNLLB! 🚀
