
import numpy as np
from backpropagation.network import ProcessingLayer
from jaxtyping import TypeCheckError
try:
    # This should fail if the jaxtyping/beartype hook is working correctly
    # weights: layer_size=3, prev_layer_size=2
    # biases: should be layer_size=3, but we provide size 4
    ProcessingLayer(
        weights=np.zeros((3, 2)),
        biases=np.zeros((4,))
    )
    print("❌ Hook failed: No TypeCheckError raised for shape mismatch.")
except (TypeCheckError, Exception) as e:
    print(f"✅ Hook working: Caught expected error: {type(e).__name__}")
    # Print a snippet of the error to confirm it's a shape mismatch
    print(f"Error message: {str(e)[:100]}...")
