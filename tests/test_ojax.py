import unittest
import typing
import dataclasses
import jax
import ojax


class TestOTree(unittest.TestCase):
    def test_warn_no_annotation(self):
        with self.assertWarns(ojax.NoAnnotationWarning):
            class MissingAnno(ojax.OTree):
                my_field = 42

            ma = MissingAnno()

    def test_field_type(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: ojax.OTree = ojax.ignore()
            d: typing.ClassVar[jax.Array]
            e: dataclasses.InitVar[str]

        ftmap = {
            'a': ojax.FieldType.AUX,
            'b': ojax.FieldType.CHILD,
            'c': ojax.FieldType.IGNORE,
        }
        self.assertEqual(len(tuple(ojax.fields(MyOTree))), 3)
        for f in ojax.fields(MyOTree):
            self.assertEqual(ojax.get_field_type(f), ftmap[f.name])

    def test_aux_infer(self):
        class MyOTree(ojax.OTree):
            a: int
            b: str = dataclasses.field(init=False)
            c: float = 3.8
            d: "typing.Any" = dataclasses.field(default=42)
            e: tuple[jax.Array] = dataclasses.field(default_factory=tuple)

        for f in ojax.fields(MyOTree):
            self.assertEqual(
                ojax.OTree.__infer_otree_field_type__(f), ojax.FieldType.AUX
            )

    def test_child_infer(self):
        class MyOTree2(ojax.OTree):
            a: jax.Array
            b: jax.numpy.ndarray
            c: ojax.OTree

        for f in ojax.fields(MyOTree2):
            self.assertEqual(
                ojax.OTree.__infer_otree_field_type__(f), ojax.FieldType.CHILD
            )

    def test_init(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: ojax.OTree = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None

            def __post_init__(self, e) -> None:
                assert e == 'lol'

        try:
            my_otree = MyOTree(a=0, b=4.2, c=None, e='lol')
        except:
            self.fail("OTree initialization failed unexpectedly.")

    def test_update(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: typing.Any = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None

            def __post_init__(self, e) -> None:
                assert e == 'lol'

        my_otree = MyOTree(a=0, b=4.2, c=None, e='lol')

        # changing children and ignored should work
        self.assertEqual(my_otree.b, 4.2)
        try:
            my_otree = my_otree.update(
                b=jax.numpy.full(1, 42.0), c=0
            )
        except:
            self.fail("OTree update failed unexpectedly.")
        self.assertEqual(my_otree.b, 42.0)
        # changing aux should fail
        with self.assertRaises(ValueError):
            my_otree.update(a=1)
        # changing child structure should fail
        with self.assertRaises(ValueError):
            my_otree.update(b=[4.0, 2.0])
        # updating undeclared field should fail
        with self.assertRaises(ValueError):
            my_otree.update(unknown_field=None)
        # updating class var should fail
        with self.assertRaises(ValueError):
            my_otree.update(d=42)
        # updating init var should fail
        with self.assertRaises(ValueError):
            my_otree.update(e='lel')

    def test_new(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: typing.Any = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None

            def __post_init__(self, e) -> None:
                assert e == 'lol'

        my_otree = MyOTree(a=0, b=4.2, c=None, e='lol')

        # changing children and ignored should work
        self.assertEqual(my_otree.b, 4.2)
        try:
            my_otree = ojax.new(
                my_otree, a=1, b=[4.0, 2.0], c=0
            )
        except:
            self.fail("ojax.new() failed unexpectedly.")
        self.assertEqual(my_otree.b, [4.0, 2.0])
        # updating undeclared field should fail
        with self.assertRaises(ValueError):
            ojax.new(my_otree, unknown_field=None)
        # updating class var should fail
        with self.assertRaises(ValueError):
            ojax.new(my_otree, d=42)
        # updating init var should fail
        with self.assertRaises(ValueError):
            ojax.new(my_otree, e='lel')

    # noinspection PyDataclass
    def test_classvar(self):
        class MyOTree(ojax.OTree):
            d: typing.ClassVar[typing.Any] = None

        my_otree = MyOTree()

        # assign from OTree instance level should fail
        with self.assertRaises(dataclasses.FrozenInstanceError):
            my_otree.d = 42
        # changing it from class level should work
        try:
            MyOTree.d = 42
        except:
            self.fail("ClassVar update failed unexpectedly.")

    def test_jax_method(self):
        class BatchMean(ojax.OTree):
            shift: float

            def __call__(self, t: jax.Array) -> jax.Array:
                return jax.vmap(lambda x: jax.numpy.mean(x) + self.shift)(t)

        bmean42 = BatchMean(42.0)
        array = jax.numpy.arange(6).reshape(2, 3)
        try:
            out = jax.jit(bmean42)(array)
        except:
            self.fail("Jitted OTree method failed.")

        self.assertTrue(
            jax.numpy.allclose(out, jax.numpy.mean(array, axis=1) + 42)
        )


if __name__ == '__main__':
    unittest.main()
