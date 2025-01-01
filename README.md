# ML Notes

##### How an AI Works SUMMARY

###### Input Based AIs

1. You have a specific `input` that you feed into the AI. This can take the form of a picture or a sentence, a number... etc.

2. (TOKENIZATION) The machine takes that `input` and converts it into numbers (also called `tokens`).

    - Tokenization is basically a way we can translate our language to the computer's language
    - Analogy: A tokenizer is quite literally a translator, if we want to say 'hello' in spanish, we enter 'hello' (`input`) into google translate (`tokenizer`) which then outputs 'hola' (`computer readable language`, in this case, a bunch of numbers, similar  to how words are just "a bunch of letters")
    Now if we reverse this process, and translate spanish back into english, entering 'hola amigo' (`computer readable language`) into google translate (`tokenizer`), we get 'hello friend' (`output`)

```
+---------+            +--------------------------------------+
|         |            | TOKENIZER                            |
|  Image  |    --->    |                                      |
|         |            |                                      |
+---------+            |  +---------+                         |
                       |  |         |         [[1, 2, 3],     |
                       |  |  Image  | ------>  [5, 2, 0],     |
                       |  |         |          [9, 4, 4]]     |
                       |  +---------+                         |
                       +--------------------------------------+

                       In this example, we create a 2-dimensional array
                       where each element represents each pixel in the image.

                       So 1  = red pixel
                          2  = blue pixel
                          ...
                          and so on


                       +--------------------------------------+
+--------+             | TOKENIZER                            |
| String |     --->    |                                      |
+--------+             |                                      |
                       |                                      |
                       |  +--------+                          |
                       |  | String | ------>  [1,4,7,2,3,5]   |
                       |  +--------+                          |
                       |                                      |
                       +--------------------------------------+

                       In this example, we enter a word, "String",
                       and the tokenizer converts each letter into
                       a number.

                       So 1 = S
                          4 = t
                          7 = r
                          ...
                          and so on

```

... TO BE CONTINUED

##### How an AI Works DETAILS

###### Tokenization

The essence of tokenization is just converting letters to numbers and numbers back into letters. We can do this by just assigning letters/characters/symbols to numbers like so:

```
? = 97
! = 98
<space-bar> = 99
a = 0
b = 1
c = 2
...
and so on
```

There's obviously more to it, like grouping by common characters like `ch`, `th` and making each group equal to a number and other fancy methods. Adding onto the previous example:
```
? = 97
! = 98
<space-bar> = 99
a = 0
b = 1
c = 2
...
ch = 56           ('ch'-aracter)
th = 57           ('th'-atch)
fr = 58           ('fr'-iend)
ght = 59          (thou-'ght')
...
and so on
```

Note that you can also group by words or some other method.

###### Between Tokenization and The Model

After the "dictionary" of tokens has been created, we can call this "dictionary" the "vocabulary" of the model. Before the model starts learning, the model will initialize a set of "contexts" for each token.

Lets say that our tokenizer converts words to numbers, and our "vocabulary" looks like the following:

```
vocabulary = {
     "Hello": 1
         "I": 2
      "like": 3
     "books": 4
       "and": 5
    "trains": 6
}
```

The model will create what we call an `embedding_layer`. Where it initializes random values into the context (we can determine the number of context) for each word (properly called, `token`). This causes the `embedding_layer` to look like the following

```
embedding_layer = {
    "1": [0.1, 0.6, 0.2, 0.1], # 1 "Hello"
    "2": [0.1, 0.1, 0.8, 0.0], # 2 "I"
    "3": [0.3, 0.2, 0.2, 0.2], # 3 "like"
    "4": [0.7, 0.1, 0.1, 0.1], # 4 "books"
    "5": [0.1, 0.3, 0.3, 0.3], # 5 "and"
    "6": [0.2, 0.3, 0.4, 0.1]  # 6 "trains"
}
```

For ease of understanding, here is a wrong but similar version the `embedding_layer`:

```
vocabulary = {
     "Hello": {"Happy": 0.1, "Angry": 0.6, "Sad": 0.2, "Neutral": 0.1}
         "I": {"Happy": 0.1, "Angry": 0.1, "Sad": 0.8, "Neutral": 0.0}
      "like": {"Happy": 0.3, "Angry": 0.2, "Sad": 0.2, "Neutral": 0.2}
     "books": {"Happy": 0.7, "Angry": 0.1, "Sad": 0.1, "Neutral": 0.1}
       "and": {"Happy": 0.1, "Angry": 0.3, "Sad": 0.3, "Neutral": 0.3}
    "trains": {"Happy": 0.2, "Angry": 0.3, "Sad": 0.4, "Neutral": 0.1}
}
```

Note that in the wrong example, it shows the context, in this case "emotion", and it's randomly initialized value (because the model does not understand "emotions" yet) for each token

The correct `embedding_layer` contains only the values.

##### The Model

###### Tensors

Tensors are literally 2-D/3-D/some_higher_dimension Array but have the properties of a matrix.
   - Indexing is the exact same as an array
   - You CANNOT append or manipulate tensors like arrays

###### Logits

Raw data generated by a model are called logits
   - To produce a logit you must have an existing model

Using the above example:
```
Given: [1,4]

Stored data in the Bigram Model
1,4 predicts 7            | 7 is a logit

So I add 7 to [1,4]       | [1,4,7]

4,7 predicts 2            | 2 is a logit

So I add 2 to [1,4,7]     | [1,4,7,2]
...
and so on until I get [1,4,7,2,3,5]

Note that the output [1,4,7,2,3,5] is NOT a logit
```

###### Batch, Time, Channels (B,T,C)

Batch is the number of inputs that are being processed by the model

Say I have a model that processes images, this is a visual representation of 3 batches

```
      +------------------+
      | Image 3          |
   +------------------+  |
   | Image 2          |  |
+------------------+  |  |
| Image 1          |  |  |
|                  |  |--+
|                  |  |
|                  |--+
|                  |
+------------------+

Think of a the above data as a cube and the Depth is Batch

      +-------+
     /       /|
    /       / |
   /       /  |
  +-------+   |
  |       |   |
  |       |   +
  |       |  /  ^
  |       | / Batch
  |       |/   v
  +-------+


Thinking simpler, you could have 5 sentences such as:
 - "Hello World!"
 - "Books are dumb"
 - "Strings"
 - "Helicopters are cars with 1 wheel"
 - "Fool Pathway"

and batch would be 5

REMEMBER that Batch is the number of inputs NOT the inputs themselves
```

Time is the maximum size of inputs

Using the above example:
```
                                                                          [ 0,
                                                                            1,
+------------------+                                                        2,
| Image 1          |                                                        3,
|                  |        [[1, 2, 3],                                     4,
|                  | ----->  [5, 2, 0], -----> [1,2,3,5,2,0,9,4,4] ----->   5,
|                  |         [9, 4, 4]]                                     6,
|                  |                                                        7,
+------------------+                                                        8 ]



If the image is too small, then the empty sections would be filled with null or similar values (but must be consistent)

In this case, Time is a 3 x 3 image, if we assume each token is 1 pixel. So we "flatten" the 2-D array into a 1-D array and Time would be 9.

Using the cube example, Time would be the height

      +-------+
     /       /|  ^
    /       / |  |
   /       /  | Time
  +-------+   |  |
  |       |   |  v
  |       |   +
  |       |  /
  |       | /
  |       |/
  +-------+

Thinking simpler, you could have 5 sentences such as:
 - "Hello World!"
 - "Books are dumb"
 - "Strings"
 - "Helicopters are cars with 1 wheel"
 - "Fool Pathway"

In this case, if the sentences are shorter than 33 characters long, then the empty sections would be filled with null or similar values.

and Time would be 33

REMEMBER that Time is the maximum size of inputs NOT the inputs themselves

Time can be thought of as the index for each pixel's data

```

Channels are the total amount of contextual information (converted into numbers) for each token and is consistent for every token.

Using the above example:
```
+------------------+
| Image 1          |
|                  |        [[1, 2, 3],
|                  | ----->  [5, 2, 0], -----> [1,2,3,5,2,0,9,4,4]
|                  |         [9, 4, 4]]
|                  |
+------------------+

Lets say the image is Grayscale.
 - Then each pixel is some shade of gray
 - So Channels = 1

Lets say the image is RGB.
 - Then each pixel is some combination of RGB values e.g. (255,123,93)
 - So Channels = 3

Using the cube example, Channels would be the length

      <Channels>
      +-------+
     /       /|
    /       / |
   /       /  |
  +-------+   |
  |       |   |
  |       |   +
  |       |  /
  |       | /
  |       |/
  +-------+


Thinking in terms of sentences, you could have 5 sentences such as:
 - "Hello World!"
 - "Books are dumb"
 - "Strings"
 - "Helicopters are cars with 1 wheel"
 - "Fool Pathway"

In this case, "Hello World!" would be {Happy: 0.8, Sad: 0.2}
              "Books are dumb" would be {Happy: 0.3, Sad: 0.7}
            etc.

and Channel would be 2

we could add more emotions
 - channel would have to increase by 1 for each emotion we add

REMEMBER that Channel is the AMOUNT of context for each token NOT the contextual information itself

REMEMBER that Channel is CONSISTENT for each token
```


Here is a visual summary

```
                  +------------------+
                  | Context          |
               +------------------+  |
               | Context          |  |
        ^   +------------------+  |  |
        |   | Context          |  |  |
        |   | for              |  |--+  ^
      Time  | Batch            |  |    /
        |   | 1                |--+  No. of Data (Batch)
        |   | (Channels)       |    /
        v   +------------------+   v
            <-----Channels----->

Using the RGB image example

                  +-----------------------+
                  | pix1: (122, 255, 156) |
               +-----------------------+  |
               | pix1: (087, 090, 000) |  |
        ^   +-----------------------+  |  |
        |   | pix1: (123, 111, 255) |  |  |
        |   | pix2: (000, 142, 224) |  |--+  ^
      Time  | pix3: (086, 039, 209) |  |    /
        |   | pix4: (090, 222, 211) |--+  No. of Data (Batch)
        |   | pix5: (255, 255, 254) |    /
        v   +-----------------------+   v
            <--------Channels------->

Note that the actual tensor for the first batch would look like this:

Tensor([[123, 111, 255],         # Each row denotes one pixel
        [000, 142, 224],         # Each column denotes the R/G/B Values for each pixel
        [086, 039, 209],
        [090, 222, 211],
        [255, 255, 254]]
        )
```

###### Single Head Self-Attention

Note that it is completely different from the standard Neural Networks (FNN, CNN, RNN).

Self-Attention is the method a model uses to determine the relationship between tokens.

At the very beginning, the model will take in the values which it will be trained on. In this case, let's assume that the input is the same as the above example, where the String is "I like books and trains".

The model will then parse this string through the tokenizer and get a list of tokens. The tokens will then be compared against the `embedding_layer` to get a list of arrays that contain the (initially random values) contexts.

```
"I like books and trains" --> [1,2,3,4,5,6] via tokenizer

embedding_layer = {
    "1": [0.1, 0.6, 0.2, 0.1], # 1 "Hello"
    "2": [0.1, 0.1, 0.8, 0.0], # 2 "I"
    "3": [0.3, 0.2, 0.2, 0.2], # 3 "like"
    "4": [0.7, 0.1, 0.1, 0.1], # 4 "books"
    "5": [0.1, 0.3, 0.3, 0.3], # 5 "and"
    "6": [0.2, 0.3, 0.4, 0.1]  # 6 "trains"
}

substituting each value in [1,2,3,4,5,6] with the respective value in the embedding_layer gives us our final output

final output:
Tensor[[0.1, 0.6, 0.2, 0.1],
       [0.1, 0.1, 0.8, 0.0],
       [0.3, 0.2, 0.2, 0.2],
       [0.7, 0.1, 0.1, 0.1],
       [0.1, 0.3, 0.3, 0.3],
       [0.2, 0.3, 0.4, 0.1]]
```
Because we are only training on 1 sentence, there is only 1 BATCH of data.
For the same reason, the maximum length of the sentence is 6 tokens, so there is 6 Time.
Based on the initialized embedding layer, where we determined that each token only has 4 context, there is 4 Channels.

Hence, `B,T,C = 1,6,4`.

Now that we have an initial tensor filled with data, each token must be able to communicate with each other.
To do this, we use the following code:

```
head_size = 2
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 2)
q = query(x) # (B, T, 2)
wei =  q @ k.transpose(-2, -1) # (B, T, 2) @ (B, 2, T) ---> (B, T, T)
```

This looks complicated but is extremely simple.
The `nn.Linear` function creates a tensor initialized with random values, of dimensions `head_size` by `C`. It does this with the intention of transforming a tensor with the dimension `C` into a tensor with the dimension `head_size` instead by applying a linear transformation and summarizing the values in `C` such that the data fits a smaller size.

Simply put, in `key = nn.Linear(C, head_size, bias=False)`, `key` contains a tensor with the dimensions `head_size` by `C` filled with random values.
The same goes for `query` and `value`.

In `k = key(x)`, it multiplies the tensor `x` (which is our `final output` tensor above) by the random values in `key` to get a `B,T,2` tensor.
Same goes for `query`.

We use randomly initialized values in `key` as it adds variation to the learning process and `k` contains the summary of "values I currently have"
Similarly, randomly initialized values in `query` adds variation to the learning process and `q` contains the summary of "values I am looking for"

By multiplying `q` and `k` in `wei =  q @ k.transpose(-2, -1)`, we get a tensor of dimensions B,T,T. This can be visualized as such:

```
Tensor(
    [["Relationship with Token 1": 0.3, "Relationship with Token 2": 0.4, "Relationship with Token 3": 0.1, ... ], # Token 1 Relation with other tokens
     ["Relationship with Token 1": 0.4, "Relationship with Token 2": 0.8, "Relationship with Token 3": 0.5, ... ], # Token 2 Relation with other tokens
     ["Relationship with Token 1": 0.1, "Relationship with Token 2": 0.7, "Relationship with Token 3": 0.3, ... ], # Token 3 Relation with other tokens
     ["Relationship with Token 1": 0.9, "Relationship with Token 2": 0.8, "Relationship with Token 3": 0.1, ... ], # Token 4 Relation with other tokens
     ["Relationship with Token 1": 0.4, "Relationship with Token 2": 0.2, "Relationship with Token 3": 0.7, ... ]] # Token 5 Relation with other tokens
)
```

But realistically it will look like this:
```
Tensor(
    [[0.3, 0.4, 0.1, ... ], # Token 1 Relation with other tokens
     [0.4, 0.8, 0.5, ... ], # Token 2 Relation with other tokens
     [0.1, 0.7, 0.3, ... ], # Token 3 Relation with other tokens
     [0.9, 0.8, 0.1, ... ], # Token 4 Relation with other tokens
     [0.4, 0.2, 0.7, ... ]] # Token 5 Relation with other tokens
)
```

Now we can see how each token can communicate/depend on/relate with each other. However, with the current relationship tensor, the model can "peek" at the answers.

Similar to how you can answer questions with the answer sheet in front of you. Thus, we need to make it such that the model cannot "peek" at the answers.

When the model generates text, it should create text based on the previous data. Hence, we need to apply the following:

```
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

`tril` generates a tensor as follows:
```
tensor([[1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        # Can continue depends on T
        [1, 1, 1, 1, 1, 1]])
```

and `masked_fill` replces the 0's with `-inf` so that the `softmax` can calculate correctly.

`wei` is the following tensor:
```
# A tensor of shape B, T, T
tensor([[1.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # row sums to 1
        [0.500, 0.500, 0.000, 0.000, 0.000, 0.000],   # row sums to 1
        [0.333, 0.333, 0.333, 0.000, 0.000, 0.000],   # ... etc.
        [0.250, 0.250, 0.250, 0.250, 0.000, 0.000],
        [0.167, 0.167, 0.167, 0.167, 0.167, 0.000],
        # Can continue depends on T
        [0.143, 0.143, 0.143, 0.143, 0.143, 0.143]])
```

Here you can see that the first row (first token) can only have a relationship with itself. Similarly, the second row (second token) has a relationship with the first and second token only. And so on.

Finally, the last step is this:

```
v = value(x) # (B, T, 2)
out = wei @ v # (B, T, 2)
```

Guess what `value` does. (Hint: Literally the same thing as `key` and `query`)
`value` is the information that is going to be carried forwards, similar to `key` but its use case is to be multiplied by the `wei` tensor to determine the final weighted representation of the input data based on the relationships between tokens.

This is the complete process for one "node"!

```
# Complete Code
head_size = 2
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 2)
q = query(x) # (B, T, 2)
wei =  q @ k.transpose(-2, -1) # (B, T, 2) @ (B, 2, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x) # (B, T, 2)
out = wei @ v # (B, T, 2)
```

As a result, we get a tensor containing the summary of contexts and the relationships between tokens

##### Multi-Head Self-Attention




#### Unorganized Information

##### Bi-gram Language Learning Model
This is arguably the most primitive yet easiest to understand model.

###### General Understanding

The essence of the Bigram Model is that it takes the previous TWO data points to predict the next data point.

From the beginning, say I enter "`String`" as input for training. The Tokenizer will convert "`String`" into `[1,4,7,2,3,5]`.

In "Training Mode" the Bigram Model trains itself by using two adjacent values to predict the next adjacent value.

```

[1,4,7,2,3,5]
 1,4,7
1,4 should predict 7

[1,4,7,2,3,5]
   4,7,2
4,7 should predict 2

[1,4,7,2,3,5]
     7,2,3
7,2 should predict 3

[1,4,7,2,3,5]
       2,3,5
2,3 should predict 5


Stored data in the Bigram Model
1,4 predicts 7
4,7 predicts 2
7,2 predicts 3
2,3 predicts 5
```

In "Usage Mode", say I want the Bigram Model to generate a word and I give it the input `St`.
The Tokenizer will convert `St` into `[1,4]`
The Bigram Model will then draw from the training:
```
Given: [1,4]

Stored data in the Bigram Model
1,4 predicts 7

So I add 7 to [1,4]       | [1,4,7]

4,7 predicts 2

So I add 2 to [1,4,7]     | [1,4,7,2]
...
and so on until I get [1,4,7,2,3,5]
```
Then the Tokenizer will then convert the output `[1,4,7,2,3,5]` into `String`.



###### How to Code

I do not understand how any of this works

```
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```


##### Other - Self Attention

Pre-requisite knowledge is Matrices





###### Simple Weighted Aggregation

The following creates a tensor/matrix filled with random values with the dimensions B by T by C, in this case, 4 by 8 by 2.

```
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
```

Using the above descriptions:
 - 4 batches of data
 - 8 tokens per batch of data
 - 2 contexts per token

The goal of simple weighted aggregation is to produce a matrix as follows:
```
# A tensor of shape B, T, T
tensor([[1.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # row sums to 1
        [0.500, 0.500, 0.000, 0.000, 0.000, 0.000],   # row sums to 1
        [0.333, 0.333, 0.333, 0.000, 0.000, 0.000],   # ... etc.
        [0.250, 0.250, 0.250, 0.250, 0.000, 0.000],
        [0.167, 0.167, 0.167, 0.167, 0.167, 0.000],
        # Can continue depends on T
        [0.143, 0.143, 0.143, 0.143, 0.143, 0.143]])
```

The first method for a simple weighted aggregation runs in O(n^2) time is as follows:

```
wei = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        wei[b,t] = torch.mean(xprev, 0)
```

The above method takes the t tokens (in the first iteration, `t = 1`, it takes the first token, in the second iteration, `t = 2`, it takes the first and second token, etc.) and stores the matrix in `xprev`.
It then calculates the mean of that matrix (`xprev`) and inserts the result into the corresponding token in `wei`.

###### Self-Attention Weighted Aggregation


