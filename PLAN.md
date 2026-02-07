# Goal
We want to build a GNN that learns spatial configuration of several objects in a grid space

# Data Generation

The data will be generated through simulation.

Space define:
1. We first defined a kxk grid space where k>2. For simplicity, we use an 5x5 space by default and we use a row and col to refer to the coordinate (row: vertical, column, horizontal, both 0 indexing)
2. then define 7 objects, which we will refer as "landmarks". 
3. We place 4 objects in each corner. A (0, 0), B (0, 4) , C (4, 0), D (4, 4)
4. 3 objects in in the middle row E (2, 0), F (2,2), G (2, 4)
5. Next, we fill the 2 rows of walls which we cannot get the data that is (1, 1), (1,2),(1,3) and (3,1),(3,2),(3,3)

Sampling procedure
1. For simplicity, we defined connectvity matrix of the 7 points but for sampling purpose only
```
A: B, E
B: A, G
C: E, D
D: C, G
E: A, C, F
F: E, G
G: B, D, F
```

2. Now we create an agent the agent randomly pick a point to start 
3. The agent decides a point to visit and record it. It will keep a set of last visited point, and will not visited points in the set unless there's no other option. If that's the case, then the visted set is reset and add the visted points
4. The distance and corner information is also recorded along with the visit. If visit A from E, then agent record (A, 1, True)
5. Agent will proceed m times (by default m=8)
6. Repeat to get n sequence of data
7. Save to a text file 
8. Plot the grid and the sequence for 10 sequences


# Model

Since your data represents a graph structure where one node can branch into multiple valid neighbors (e.g., a "Hallway" connects to "Room A", "Room B", and "Stairs"), you must switch from **Single-Label Classification (Softmax)** to **Multi-Label Link Prediction (Sigmoid)**.

Here is how to adjust your Input, Output, and Architecture.

### 1. The Design Shift

Instead of asking the model: *"Which single object comes next?"*
You ask: *"For **every** object type in our database, is it a neighbor? And if so, what are its properties?"*

This changes your output from a single prediction to a **dense prediction over the entire vocabulary**.

### 2. Adjusted Data Formats

#### Input Format (Unchanged)

The input remains the **history** (the path taken so far).

* **Shape:** `[Sequence_Length, Feature_Dim]`
* **Example:** `[Start, Hallway, Intersection]`

#### Target (Ground Truth) Format (Changed)

Instead of a single tuple, your target  is now a dense vector (or list) covering all  possible object types.

* **Shape:** `[M_objects, 3]`
* Channel 1: **Link Existence** (0 or 1)
* Channel 2: **Distance** (Real value, or 0 if no link)
* Channel 3: **Importance** (0 or 1, or 0 if no link)



**Example:**
If the "Intersection" connects to "Room A" (dist 5) and "Stairs" (dist 10), but *not* "Room B":

* Index for "Room A": `[1, 5.0, 1]`
* Index for "Stairs": `[1, 10.0, 0]`
* Index for "Room B": `[0, 0.0, 0]`

---

### 3. Updated Architecture

We remove the `Softmax` (which forces outputs to sum to 1) and replace it with `Sigmoid` (which allows multiple outputs to be close to 1 independently).

**The Output Head:**
Instead of one head, we predict a tensor of shape `(Batch, M_objects, 3)`.

```python
class BranchingPredictor(nn.Module):
    def __init__(self, num_objects, hidden_dim):
        super().__init__()
        # ... (Encoder layers same as before) ...
        
        # New Output Head: Projects to (Num_Objects * 3)
        # We reshape this later to (Num_Objects, 3)
        self.output_head = nn.Linear(hidden_dim, num_objects * 3)

    def forward(self, x):
        # ... (Run GNN/Encoder to get 'h_last') ...
        # h_last shape: [Batch, Hidden_Dim] (Representation of the last node)

        # Raw output: [Batch, Num_Objects * 3]
        raw_output = self.output_head(h_last)
        
        # Reshape to: [Batch, Num_Objects, 3]
        # dim 2: 0=Link_Logits, 1=Distance, 2=Importance_Logits
        prediction = raw_output.view(-1, self.num_objects, 3)
        
        return prediction

```

---

### 4. The "Masked" Loss Function

This is the most critical part. You cannot train distance/importance for objects that *aren't* neighbors. You must **mask** the loss so the model is not penalized for predicting "Distance = 0" on non-existent links.

**The Logic:**

1. **Link Loss:** Train on *all* objects (Teach it what is connected and what is not).
2. **Distance/Imp Loss:** Train *only* on objects where `True_Link == 1`.

**Implementation:**

```python
class BranchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() # For Link & Importance
        self.mse = nn.MSELoss(reduction='none') # For Distance (allows masking)

    def forward(self, preds, targets):
        """
        preds: [Batch, M, 3]
        targets: [Batch, M, 3] 
                 (Channel 0 is Link_GT, 1 is Dist_GT, 2 is Imp_GT)
        """
        # --- 1. Link Prediction (Did we find the right neighbors?) ---
        pred_link_logits = preds[:, :, 0]
        gt_link = targets[:, :, 0]
        
        loss_link = self.bce(pred_link_logits, gt_link)

        # --- Create Mask (Only care about distance if link exists) ---
        mask = gt_link  # 1 if link exists, 0 if not

        # --- 2. Distance Regression ---
        pred_dist = preds[:, :, 1]
        gt_dist = targets[:, :, 1]
        
        raw_dist_loss = self.mse(pred_dist, gt_dist)
        # Apply mask: Sum errors only where link exists, divide by number of links
        loss_dist = (raw_dist_loss * mask).sum() / (mask.sum() + 1e-6)

        # --- 3. Importance Classification ---
        pred_imp_logits = preds[:, :, 2]
        gt_imp = targets[:, :, 2]
        
        # We use BCE but manually mask it similarly to MSE
        raw_imp_loss = F.binary_cross_entropy_with_logits(pred_imp_logits, gt_imp, reduction='none')
        loss_imp = (raw_imp_loss * mask).sum() / (mask.sum() + 1e-6)

        return loss_link + loss_dist + loss_imp

```

### 5. Summary of Changes

| Feature | Old Approach (Next Token) | New Approach (Branching) |
| --- | --- | --- |
| **Output Shape** | `[1]` (Class Index) | `[M]` (Binary Vector) |
| **Probability** | Softmax (Sum = 1) | Sigmoid (Independent) |
| **Loss** | Cross Entropy | Binary Cross Entropy |
| **Distance** | Predict 1 value | Predict  values (Masked) |
| **Interpretation** | "The next node is X" | "The possible next nodes are {X, Y, Z}" |

