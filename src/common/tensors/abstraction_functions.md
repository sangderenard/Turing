# Abstract Tensor API Functions

This document groups available methods by theme for quick reference.

## Shape Accessors

## Creation & Initialization
- AbstractTensor.full()
- AbstractTensor.zeros()
- AbstractTensor.arange()
- AbstractTensor.clone()

## Device & Dtype Management
- AbstractTensor.to_device()
- AbstractTensor.get_device()
- AbstractTensor.get_dtype()
- AbstractTensor.to_dtype()
- AbstractTensor.long_dtype()
- AbstractTensor.bool_dtype()
- AbstractTensor.float_dtype()
- AbstractTensor.tensor_type()
- AbstractTensor.long_cast()
- AbstractTensor.float()
- AbstractTensor.double()
- AbstractTensor.int()
- AbstractTensor.long()
- AbstractTensor.bool()
- AbstractTensor.to_backend()
- AbstractTensor.get_tensor()

## Tensor Properties
- AbstractTensor.item()
- AbstractTensor.numel()
- AbstractTensor.shape()
- AbstractTensor.shape_()
- AbstractTensor.ndim()
- AbstractTensor.dim()
- AbstractTensor.ndims()
- AbstractTensor.datastring()
- AbstractTensor.__repr__()

## Indexing & Selection
- AbstractTensor.select_by_indices()
- AbstractTensor.index_select()
- AbstractTensor.boolean_mask_select()
- AbstractTensor.argmin()
- AbstractTensor.assign_at_indices()
- AbstractTensor.copyto()
- AbstractTensor.increment_at_indices()
- AbstractTensor.__getitem__()
- AbstractTensor.__setitem__()
- AbstractTensor.unravel_index()

## Reshaping & Manipulation
- AbstractTensor.view_flat()
- AbstractTensor.repeat()
- AbstractTensor.repeat_interleave()
- AbstractTensor.stack()
- AbstractTensor.cat()
- AbstractTensor.pad()
- AbstractTensor.clamp()
- AbstractTensor.topk()
- AbstractTensor.interpolate()

## Arithmetic & Comparison
- AbstractTensor._apply_operator()
- AbstractTensor.__add__()
- AbstractTensor.__sub__()
- AbstractTensor.__mul__()
- AbstractTensor.__truediv__()
- AbstractTensor.__floordiv__()
- AbstractTensor.__mod__()
- AbstractTensor.__pow__()
- AbstractTensor.__matmul__()
- AbstractTensor.__radd__()
- AbstractTensor.__rsub__()
- AbstractTensor.__rmul__()
- AbstractTensor.__rtruediv__()
- AbstractTensor.__rfloordiv__()
- AbstractTensor.__rmod__()
- AbstractTensor.__rpow__()
- AbstractTensor.__rmatmul__()
- AbstractTensor.__iadd__()
- AbstractTensor.__isub__()
- AbstractTensor.__imul__()
- AbstractTensor.__itruediv__()
- AbstractTensor.__ifloordiv__()
- AbstractTensor.__imod__()
- AbstractTensor.__ipow__()
- AbstractTensor.__imatmul__()
- AbstractTensor.not_equal()
- AbstractTensor.less()
- AbstractTensor.pow()
- AbstractTensor.sqrt()
- AbstractTensor.mean()
- AbstractTensor.cumsum()
- AbstractTensor.max()
- AbstractTensor.log_softmax()

## Trigonometric & Hyperbolic
- AbstractTensor.sin()
- AbstractTensor.cos()
- AbstractTensor.tan()
- AbstractTensor.asin()
- AbstractTensor.acos()
- AbstractTensor.atan()
- AbstractTensor.sinh()
- AbstractTensor.cosh()
- AbstractTensor.tanh()
- AbstractTensor.asinh()
- AbstractTensor.acosh()
- AbstractTensor.atanh()
- AbstractTensor.sec()
- AbstractTensor.csc()
- AbstractTensor.cot()
- AbstractTensor.sech()
- AbstractTensor.csch()
- AbstractTensor.coth()
- AbstractTensor.sinc()

## Spectral Transforms
- AbstractTensor.fft()
- AbstractTensor.ifft()

## Persistence
- AbstractTensor.save()
- AbstractTensor.load()

## Utilities
- AbstractTensor.benchmark()
- AbstractTensor.data_or()
- AbstractTensor.get_shape()
- AbstractTensor.get_ndims()
- AbstractTensor.ensure_tensor()

## Functional Interface
- AbstractF.interpolate()

## Module-Level Helpers
- register_conversion()
- _get_ops_for_class()
- _find_conversion_path()
- _get_shape()
- _flatten()
- default_to_backend()
- get_tensor_operations()

## Missing or Incomplete Functions
- AbstractTensor.__setitem__ for CTensor backend
- c_backend.repeat_interleave_
- c_backend.assign_at_indices_
- c_backend.increment_at_indices_
- c_backend.boolean_mask_select_
- c_backend.index_select_
- c_backend.argmin_
- c_backend.interpolate_
- c_backend.stack_
- c_backend.cat_
- jax_backend._apply_operator__
- torch_backend._apply_operator__
- numpy_backend._apply_operator__
- pure_backend._apply_scalar_op
- pure_backend._matmul
- pure_backend.cat_
- pure_backend.repeat_interleave_
- pure_backend.mean_
- pure_backend.index_select_
