# Data Directory

Place datasets here. CIFAKE-style layouts are supported:

```text
data/cifake/train/REAL/*.jpg
data/cifake/train/FAKE/*.jpg
data/cifake/test/REAL/*.jpg
data/cifake/test/FAKE/*.jpg
```

Generic layouts also work as long as a parent directory contains a recognizable label such as `real`, `fake`, `authentic`, `synthetic`, or `generated`.
