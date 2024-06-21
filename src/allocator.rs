#[cfg(all(
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
use tikv_jemallocator::Jemalloc;
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
use mimalloc::MiMalloc;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "mimalloc"),
    not(allocator = "default"),
    target_family = "unix",
))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
static ALLOC: MiMalloc = MiMalloc;