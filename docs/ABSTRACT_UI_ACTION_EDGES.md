# AbstractUI system timer and action edges

Every browser document assembled from an `AbstractUI` object requires a
JavaScript system-root prelude. The prelude creates one timer, one pending
action queue, and a connection point for the system-root action-edge table.

```text
system root
  timer
    -- update(actions) -->
  entity mezzanine
    action-edge table
      row: source, interaction, destination, count, last-issued, recent
```

Interactive nodes continue to declare only interaction type and destination.
During projection, each node gets an action-edge row whose identity also
includes the source node. The omnipotent event host issues a record containing
actor, type, destination, edge identity, and issue time. It does not alter the
row.

On each system frame the timer calls exactly:

```javascript
actionEdges.update(actions)
```

The table increments affected rows and lights rows whose last issue occurred
inside the configured recent-event window. Empty timer deliveries extinguish
expired rows without erasing their counts or history timestamps.

The timer-to-table relation and operation spelling also exist in the neutral
Python model in `abstract_ui_actions.py`. The browser's animation frame is one
timer backend. A fixed simulation clock, audio clock, SDL loop, worker, or
replay log can deliver the same action batches without changing edge identity.
