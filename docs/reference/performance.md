# Performance measurements

Throughput depends on the checkpoint, input lengths, number of questions and
options, hardware and scheduling settings. Report a workload alongside a speed
rather than presenting one benchmark as a universal package rate.

## Measure the complete call

Warm up with a separate input, then time `.query(requests)` or the complete group
of `.aquery()` submissions until typed results are returned. This includes
encoding, submission, model execution and decoding. Report startup separately.
For diagnosis, measure those components individually without confusing component
times with end-to-end throughput.

Record the package and model versions, GPU count and model, precision, memory
utilization, token budget, maximum sequences, batch size and active concurrency.
For backend comparisons, match effective settings and repeat on the same device.
State whether prefix caching and chunked prefill are enabled.

Check how many states and query texts are distinct, and do not prefill the timed
workload during warmup. Prefix sharing between options for one document is normal
classification work and should be reported rather than disabled silently.

## Count work consistently

One classification means one question evaluated over its candidate answers. A
request with five questions produces five classifications. Report both document
and question counts if those differ.

Submitted candidate tokens repeat the document and question prefix for each
option. They are not the same as unique document tokens, nor the tokens actually
computed after prefix-cache reuse. Label the token count explicitly; do not call
summed candidate lengths uncached input throughput.

## Convert a rental rate to cost

For a rental rate `R` in dollars per hour and measured classification throughput
`C` per second:

```text
Dollars per million classifications = R × 1,000,000 / (3,600 × C)
```

For a precisely defined token throughput `T` per second, the same formula with
`T` gives dollars per million of those tokens. Publish the provider, GPU SKU,
region, rate date, rate source and whether the rate is on-demand, spot or reserved.

These formulas estimate steady-state inference cost, not the cost of a short job
including startup, idle time and minimum billing. Use the resources actually
charged: an Alps whole-node reservation cannot be priced as one rented GPU merely
because only one GPU was made visible to the model.
