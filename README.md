# Vision-Transformer-ViT-in-Tensorflow
Implementation of Vision Transformer, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Tensorflow
## Usage
```python
class PreNorm(layers.Layer):
  def __init__(self, fn):
    super(PreNorm, self).__init__()
    self.norm = layers.LayerNormalization(epsilon=1e-6)
    self.fn = fn

  #@tf.function(jit_compile=True)  
  def call(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)
```

```python
class FeedForward(layers.Layer):
  
  def __init__(self, dim, hidden_dim, dropout=0.1):
    super(FeedForward, self).__init__()
    self.net =  keras.Sequential([
                    layers.Dense(hidden_dim, activation=tf.nn.gelu),
                    #tfa.layers.GELU(),
                    layers.Dropout(dropout),
                    layers.Dense(dim),
                    layers.Dropout(dropout)
                    ])
  #@tf.function(jit_compile=True)
  def call(self, x):
    return self.net(x)
```

```python
class Attention(layers.Layer):

  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
     
    super(Attention, self).__init__()
    self.heads = heads
    self.dim_head = dim_head

    self.inner_dim = self.dim_head *  self.heads

    self.to_q = layers.Dense(self.inner_dim)
    self.to_k = layers.Dense(self.inner_dim)
    self.to_v = layers.Dense(self.inner_dim)

    self.scale = 1/K.sqrt(K.cast(dim_head, 'float32'))
    self.attend = layers.Activation('softmax')
    self.to_out = keras.Sequential([
            layers.Dense(dim),
            layers.Dropout(dropout)])
   
  #@tf.function(jit_compile=True)
  def call(self, inputs):
    batch_size = K.int_shape(inputs)[0] #tf.shape(inputs)[0]

    q = self.to_q(inputs)
    k = self.to_k(inputs)
    v = self.to_v(inputs)

    q = K.reshape(q, (batch_size, -1, self.heads, self.dim_head))
    k = K.reshape(k, (batch_size, -1, self.heads, self.dim_head))
    v = K.reshape(v, (batch_size, -1, self.heads, self.dim_head))

    q = K.permute_dimensions(q, (0, 2, 1, 3))
    k = K.permute_dimensions(k, (0, 2, 1, 3))
    v = K.permute_dimensions(v, (0, 2, 1, 3))

    dots = tf.matmul(q, k, transpose_b=True) * self.scale
    attn = self.attend(dots)

    out = tf.matmul(attn, v)
    out = K.permute_dimensions(out, (0, 2, 1, 3))
    out = K.reshape(out, (batch_size, -1, self.inner_dim))
    
    return self.to_out(out)
```

```python
class Transformer(layers.Layer):
  
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super(Transformer, self).__init__()
    self.layers = []
    for _ in range(depth):
      self.layers.append(
          [PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
           PreNorm(FeedForward(dim, mlp_dim, dropout = dropout))])
  
  #@tf.function(jit_compile=True)        
  def call(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return x
```
```python
def pair(t):
  return t if isinstance(t, tuple) else (t, t) 

class ViT(layers.Layer):
  
  def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
               pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    
    super(ViT, self).__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

    self.num_patches = (image_height // patch_height) * (image_width // patch_width)
    self.patch_dim   = channels * patch_height * patch_width
    assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
    
    self.patch_size = patch_size
    self.dim = dim
    self.dense = layers.Dense(self.dim)

    self.pos_embedding = self.add_weight(shape=[1, self.num_patches+1, self.dim],dtype=tf.float32) 
    self.cls_token = self.add_weight(shape=[1, 1, self.dim],dtype=tf.float32) 
    self.dropout = layers.Dropout(emb_dropout)

    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    self.pool = pool
    self.to_latent1 = layers.Dropout(0.1)
    self.to_latent2 = layers.Dense(mlp_dim, activation=tfa.activations.gelu)

    self.mlp_head = keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)]) #, activation='softmax'
      
  def build(self, input_shape):
    self.b = input_shape[0]
    super(ViT, self).build(input_shape)
    
  #@tf.function(jit_compile=True)
  def call(self, inputs):
    
    x = tf.nn.space_to_depth(inputs, self.patch_size)
    x = K.reshape(x, (-1, self.num_patches, self.patch_dim))
    x = self.dense(x)
    b = tf.shape(x)[0] #b , _ , _ = x.shape
    
    cls_tokens = tf.repeat(self.cls_token, b, axis=0)
    x = tf.concat((cls_tokens, x), axis=1)
    
    pos_emb = tf.repeat(self.pos_embedding, b, axis=0)  
    
    x += pos_emb
    x = self.dropout(x)

    x = self.transformer(x)

    x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

    x = self.to_latent1(self.to_latent2(x))
    return self.mlp_head(x)
```


```python
from tensorflow.keras import backend as K

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 8,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
model.build(input_shape=(1,256,256,3))


img = tf.random.uniform(shape=[1, 256, 256, 3])
preds = model(img)
print(preds.shape) # (1, 1000)
```
