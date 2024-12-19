#!/bin/bash

for i in {1..5}
do
  {
    echo "Processing item $i..."
    sleep 5  # Simulate some work with a random delay
    echo "Finished processing item $i."
  } &
done

# Wait for all background processes to complete
wait

echo "All tasks are complete. Proceeding to the next step."
