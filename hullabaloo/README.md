# hullabaloo

Backend-agnostic geometry construction utilities for convex polytope families.

## Features

- **Prismatoids**: Construct vertex sets for drums and related families.
- **Generic over `Num`**: No hidden `f64` intermediates---precision preserved throughout.
- **Explicit conversion**: Generate vertices, then convert to your backend's matrix type.

## Example

```rust
use hullabaloo::{Drum, Geometrizable};

let top = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
let bot = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0]];
let drum = Drum::<f64>::new(top, bot)?;
let vertices = drum.into_vertices();

assert_eq!(vertices.len(), 6);
```

## License

AGPL-3.0-only. See `LICENSE` for details.
