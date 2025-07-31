# Python API Proposal for ext_py (WIP)

This document outlines the proposed Python API for the `ext_py` crate, based on translations of the
existing Rust examples in the `ext` crate.

## Core Module Structure

```python
import ext_py
```

## Initialization

```python
ext_py.init_logging()  # Initialize logging system
```

## Algebras

### Milnor Algebra

```python
algebra = ext_py.MilnorAlgebra(prime=2, truncated=False)
algebra.compute_basis(max_degree)
dim = algebra.dimension(degree)
```

### Adem Algebra  

```python
algebra = ext_py.AdemAlgebra(prime=2, truncated=False)
algebra.compute_basis(max_degree)
generators = algebra.generators(degree)
basis_string = algebra.basis_element_to_string(degree, index)
```

### Steenrod Algebra (Union type)

```python
algebra = ext_py.SteenrodAlgebra.adem_algebra(prime=2, truncated=False)
algebra = ext_py.SteenrodAlgebra.milnor_algebra(prime=2, truncated=False)
```

## Modules

### Finite Dimensional Module

```python
module = ext_py.FDModule(algebra, name, graded_dims, min_degree=0)
module.set_basis_element_name(degree, index, name)
module.set_action(op_degree, op_index, input_degree, input_index, output_vector)
module.extend_actions(input_degree, output_degree)
module.check_validity(input_degree, output_degree)
module.to_json()  # Returns dict

# Querying module properties
dim = module.dimension(degree)
max_deg = module.max_computed_degree()
min_deg = module.min_degree()
name = module.basis_element_name(degree, index)
```

### Tensor Module

```python
tensor = ext_py.TensorModule(left_module, right_module)
fd_tensor = ext_py.FDModule.from_tensor_module(tensor)
```

### Module Loading

```python
module_json = ext_py.parse_module_name("S_2")  # Returns dict
module = ext_py.steenrod_module_from_json(algebra, module_json)
```

## Coordinates and Bidegrees

```python
bidegree = ext_py.Bidegree.from_t_s(t, s)
bidegree = ext_py.Bidegree.from_n_s(n, s)
t_value = bidegree.t
s_value = bidegree.s

generator = ext_py.BidegreeGenerator(bidegree, index)
```

## Resolutions and Chain Complexes

### Resolution

```python
resolution = ext_py.query_module(algebra_type=None, save=False)
resolution = ext_py.query_module_only(prompt, algebra_type, save)

resolution.compute_through_bidegree(max_bidegree)
resolution.compute_through_degree(max_degree)

module = resolution.module(degree)
chart_string = resolution.graded_dimension_string()
sseq = resolution.to_sseq()

# Filtration one products
products = resolution.algebra().default_filtration_one_products()
product_data = resolution.filtration_one_products(op_degree, op_index)
```

### Secondary Resolution

```python
secondary = ext_py.SecondaryResolution(resolution)
secondary.compute_partial(s)  # For distributed computation
secondary.extend_all()
homotopy = secondary.homotopy(s)
underlying = secondary.underlying()
```

### Unstable Resolutions

```python
resolution = ext_py.query_unstable_module(name, save=False)
resolution.compute_through_degree(max_degree)
sseq = resolution.to_unstable_sseq()
```

## Steenrod Evaluator

```python
evaluator = ext_py.SteenrodEvaluator(prime)
result = evaluator.evaluate_module_adem(expression_string)
adem_string = evaluator.adem_element_to_string(degree, element)
milnor_element = evaluator.adem_to_milnor(degree, adem_element)
milnor_string = evaluator.milnor_element_to_string(degree, element)
```

## Spectral Sequences and Charts

### Spectral Sequence

```python
sseq = resolution.to_sseq()
sseq = resolution.to_unstable_sseq()

sseq.write_to_graph(
    backend=svg_backend,
    page_number=2,
    show_differentials=False,
    products=products,
    callback=lambda x: None
)

sseq.write_unstable_chart(
    backend=svg_backend,
    max_degree=max_degree,
    show_differentials=True
)
```

### SVG Backend

```python
import sys
svg_backend = ext_py.SvgBackend(sys.stdout)
```

## Higher Operations

### Massey Products

```python
massey_computer = ext_py.MasseyProductComputer(resolution)
result = massey_computer.compute_massey_product(elements_list)

# Result methods
is_zero = result.is_zero()
is_indeterminate = result.is_indeterminate()
indeterminacy = result.indeterminacy()
```

## Algebra Types (Enum)

```python
ext_py.AlgebraType.MILNOR
ext_py.AlgebraType.ADEM
```

## Vector Operations

```python
# Module elements and operations
result = module.apply_operation(operation_string, element_degree, element_index)
element_string = module.element_to_string(degree, vector)
is_zero = vector.is_zero()
```

## Iterator Support

```python
# Iteration over bidegrees
for bidegree in resolution.iter_nonzero_stem():
    # Process bidegree
    pass

# Iteration over homotopy maps
for i, entry in enumerate(homotopy_map):
    # Process entry
    pass
```

## Error Handling

All functions that can fail should raise appropriate Python exceptions, following standard Python
error handling patterns.

## Interactive Utilities

The API should support interactive querying similar to the Rust version:

```python
# These would be internal utilities used by the examples
choice = ext_py.query.with_default(prompt, default, parser_function)
yes_no = ext_py.query.yes_no(prompt)
optional_value = ext_py.query.optional(prompt, parser_function)
raw_input = ext_py.query.raw(prompt, validator_function)
```

## Notes on Design

1. **Pythonic Naming**: Function and method names follow Python conventions (snake_case)
2. **Type Safety**: Consider using type hints throughout the API
3. **Memory Management**: PyO3 handles Rust ↔ Python memory management
4. **Error Propagation**: Rust errors should be converted to appropriate Python exceptions
5. **Iterator Protocol**: Implement Python iterator protocol for Rust iterators
6. **Context Managers**: Consider implementing context managers for resources that need cleanup
7. **JSON Serialization**: Support standard Python dict/JSON serialization patterns
8. **Default Arguments**: Use Python default arguments where appropriate
9. **Operator Overloading**: Consider implementing `__add__`, `__sub__`, etc. for mathematical
   objects

## Implementation Priority

Based on the examples translated, the following components should be prioritized:

1. **Core algebras** (Milnor, Adem)
2. **Basic module types** (FDModule, tensor products)
3. **Resolution computation**
4. **Coordinate systems** (Bidegree, BidegreeGenerator)
5. **Chart generation**
6. **Interactive utilities**
7. **Higher operations** (secondary, Massey products)
8. **Unstable computations**

This API design balances staying true to the Rust implementation while providing a natural Python
interface that follows Python conventions and idioms.
