from collections import namedtuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from e3nn_jax.utils import assert_equivariant

from mace_jax import modules
from mace_jax.modules import MACE, SymmetricContraction


def test_symmetric_contraction():
    x = e3nn.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 4))

    model = hk.without_apply_rng(
        hk.transform(
            lambda x, y: SymmetricContraction(3, ["0e", "1o", "2e"])(x, y)
        )
    )
    w = model.init(jax.random.PRNGKey(2), x, y)

    assert_equivariant(
        lambda x: model.apply(w, x, y), jax.random.PRNGKey(3), args_in=(x,)
    )


# TODO fix this test
def test_mace():
    atomic_energies = np.array([1.0, 3.0], dtype=float)

    @hk.without_apply_rng
    @hk.transform
    def model(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes, #scalar_features]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        return MACE(
            r_max=2.0,
            radial_basis=lambda r, r_max: e3nn.bessel(r, 8, r_max),
            radial_envelope=lambda r, r_max: e3nn.poly_envelope(5 - 1, 2, r_max)(r),
            max_ell=3,
            num_interactions=2,
            num_species=1,
            hidden_irreps="11x0e+11x1o",
            readout_mlp_irreps="16x0e",
            avg_num_neighbors=3.0,
            correlation=2,
            output_irreps="0e",
            symmetric_tensor_product_basis=False,
        )(vectors, node_specie, senders, receivers)

    Node = namedtuple("Node", ["positions", "attrs"])
    Edge = namedtuple("Edge", ["shifts"])
    Globals = namedtuple("Globals", ["cell"])

    graph = jraph.GraphsTuple(
        nodes=Node(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            attrs=jnp.array([0, 1]),
        ),
        edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        globals=Globals(cell=jnp.eye(3) * 3),
        senders=jnp.array([0, 1]),
        receivers=jnp.array([1, 0]),
        n_edge=jnp.array([2]),
        n_node=jnp.array([2]),
    )
    vectors = (
        graph.nodes.positions[graph.receivers] + graph.edges.shifts @ graph.globals.cell
    ) - graph.nodes.positions[graph.senders]

    w = model.init(
        jax.random.PRNGKey(0), 
        vectors,
        graph.nodes.attrs,
        graph.senders,
        graph.receivers
    )

    def wrapper(positions):
        graph = jraph.GraphsTuple(
            nodes=Node(
                positions=positions.array,
                attrs=jnp.array([0, 1]),
            ),
            edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
            globals=Globals(cell=jnp.eye(3) * 3),
            senders=jnp.array([0, 1]),
            receivers=jnp.array([1, 0]),
            n_edge=jnp.array([2]),
            n_node=jnp.array([2]),
        )
        vectors = (
            graph.nodes.positions[graph.receivers] + graph.edges.shifts @ graph.globals.cell
        ) - graph.nodes.positions[graph.senders]

        energy = model.apply(
            w, 
            vectors,
            graph.nodes.attrs,
            graph.senders,
            graph.receivers
        )
        return e3nn.IrrepsArray("0e", energy)

    positions = e3nn.normal("1o", jax.random.PRNGKey(1), (2,))
    print(wrapper(positions))
    # assert_equivariant(wrapper, jax.random.PRNGKey(1), positions)


if __name__ == "__main__":
    test_mace()
    # test_symmetric_contraction()
