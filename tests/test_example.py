from dataclasses import field
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split as jrsplit, normal as jrnormal
from ojax import aux, child, OTree, fields


# defines a fully connected layer for neural networks
class Dense(OTree):
    input_features: int  # inferred to be auxiliary data
    output_features: int = aux()  # or explicit declaration
    weight: jnp.ndarray = field(default=..., init=False)  # inferred as child
    bias: jnp.ndarray = child(default=..., init=False)  # explicit declaration

    # use .assign_ only in __init__ function
    def __init__(self, input_features: int, output_features: int):
        self.assign_(
            input_features=input_features, output_features=output_features
        )

    # forward pass
    def forward(self, input_array):
        return jnp.inner(input_array, self.weight) + self.bias

    # set new parameters, notice it returns an updated version of itself
    def update_parameters(self, weight, bias):
        assert weight.shape == (self.output_features, self.input_features)
        assert bias.shape == (self.output_features,)
        return self.update(weight=weight, bias=bias)


# example usage
if __name__ == "__main__":
    # define data
    data_count, data_features, output_features = 4, 3, 2
    key = PRNGKey(0)
    key, key_data, key_weight, key_bias = jrsplit(key, 4)
    input_data = jrnormal(key_data, shape=(data_count, data_features))
    # define layer
    init_weight = jrnormal(key_weight, shape=(output_features, data_features))
    init_bias = jrnormal(key_bias, shape=(output_features,))
    layer = Dense(data_features, output_features)
    # No inplace update, need to get the returned updated layer instance!
    layer = layer.update_parameters(weight=init_weight, bias=init_bias)
    for f in fields(layer):
        print(f.name, type(f), OTree.__infer_otree_field_type__(f))
        # input_features <class 'dataclasses.Field'> <class 'ojax.otree.Aux'>
        # output_features <class 'ojax.otree.Aux'> <class 'ojax.otree.Aux'>
        # weight <class 'dataclasses.Field'> <class 'ojax.otree.Child'>
        # bias <class 'ojax.otree.Child'> <class 'ojax.otree.Child'>
    # use layer as a pytree
    layer_w, layer_b = jax.tree.flatten(layer)[0]
    assert (layer_w == init_weight).all() and (layer_b == init_bias).all()
    # flatten and unflatten recovers the layer
    layer = jax.tree.unflatten(*jax.tree.flatten(layer)[::-1])
    # compute output, notice that jax.jit / jax.vmap works out of the box
    output = jax.jit(jax.vmap(layer.forward))(input_data)
    print(output)
    # [[-2.666112   -1.0220472 ]
    #  [-3.701102   -0.8207982 ]
    #  [-4.4596996   0.6687442 ]
    #  [ 0.92416656 -3.302886  ]]
