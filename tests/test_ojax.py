import unittest
import typing
import abc
import warnings
import io
import dataclasses
import jax
import ojax


class TestOTree(unittest.TestCase):
    def test_warn_no_annotation(self):
        with self.assertWarns(ojax.NoAnnotationWarning):
            class MissingAnno(ojax.OTree):
                my_field = 42

            _ = MissingAnno()

    def test_warn_no_annotation_abc(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            class AbstractTree(ojax.OTree, metaclass=abc.ABCMeta):
                @abc.abstractmethod
                def my_method(self):
                    raise NotImplementedError

            class MyTree(AbstractTree):
                def my_method(self):
                    return 42

            mt = MyTree()
            self.assertEqual(mt.my_method(), 42)

    def test_warn_no_annotation_property(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            class WithProperty(ojax.OTree):
                @property
                def my_property(self):
                    return 42

            _ = WithProperty()

    def test_bad_default(self):
        with self.assertRaises(ValueError):
            _ = ojax.aux(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojax.child(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojax.ignore(default=(), default_factory=tuple)
        with self.assertRaises(ValueError):
            _ = ojax.alien(default=(), default_factory=tuple)

    def test_field_type(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: ojax.OTree = ojax.ignore()
            d: typing.ClassVar[jax.Array]
            e: dataclasses.InitVar[str]
            f: io.BytesIO = ojax.alien()

        ftmap = {
            'a': ojax.Aux,
            'b': ojax.Child,
            'c': ojax.Ignore,
            'f': ojax.Alien,
        }
        self.assertEqual(len(tuple(dataclasses.fields(MyOTree))), 4)
        for f in dataclasses.fields(MyOTree):
            self.assertTrue(isinstance(f, ftmap[f.name]))
        for f in ojax.fields(MyOTree):
            self.assertTrue(isinstance(f, ftmap[f.name]))
        for f in ojax.fields(MyOTree, ojax.Ignore, infer=False):
            self.assertTrue(isinstance(f, ftmap[f.name]))
        with self.assertRaises(ValueError):
            _ = ojax.fields(MyOTree, dataclasses.Field)

    def test_aux_infer(self):
        class MyOTree(ojax.OTree):
            a: int
            b: str = dataclasses.field(init=False)
            c: float = 3.8
            d: "typing.Any" = dataclasses.field(default=42)
            e: tuple[jax.Array] = dataclasses.field(default_factory=tuple)

        for f in dataclasses.fields(MyOTree):
            self.assertEqual(
                ojax.OTree.__infer_otree_field_type__(f), ojax.Aux
            )

    def test_child_infer(self):
        class MyOTree2(ojax.OTree):
            a: jax.Array
            b: jax.numpy.ndarray
            c: ojax.OTree

        for f in dataclasses.fields(MyOTree2):
            self.assertEqual(
                ojax.OTree.__infer_otree_field_type__(f), ojax.Child
            )

    def test_init(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: ojax.OTree = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None
            f: io.BytesIO = ojax.alien(default_factory=io.BytesIO)

            def __init__(self, a, b, c, e) -> None:
                assert e == 'lol'
                self.assign_(a=a, b=b, c=c)
        try:
            _ = MyOTree(a=0, b=4.2, c=None, e='lol')
        except:
            self.fail("OTree initialization failed unexpectedly.")

    def test_flatten_unflatten(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: typing.Any = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None

            def __init__(self, a, b, c, e) -> None:
                assert e == 'lol'
                self.assign_(a=a, b=b, c=c)

        my_otree = MyOTree(a=0, b=4.2, c=None, e='lol')
        try:
            vals, treedef = jax.tree.flatten(my_otree)
        except:
            self.fail("OTree flatten failed unexpectedly.")
        try:
            new_otree = jax.tree.unflatten(treedef, vals)
        except:
            self.fail("OTree unflatten failed unexpectedly.")
        for n in ('a', 'b'):
            self.assertEqual(getattr(my_otree, n), getattr(new_otree, n))

        class AlienOTree(MyOTree):
            f: io.BytesIO = ojax.alien()

            def __init__(self, a, b, c, e, f=io.BytesIO()) -> None:
                super().__init__(a, b, c, e)
                self.assign_(f=f)

            def add_b(self, val):
                return val + self.b

        alien_otree = AlienOTree(a=0, b=4.2, c=None, e='lol')
        self.assertEqual(jax.jit(alien_otree.add_b)(10), 14.2)
        with self.assertRaises(ojax.AlienException):
            _ = jax.tree.flatten(alien_otree)
        alien_otree = alien_otree.update(f=None)
        try:
            vals, treedef = jax.tree.flatten(alien_otree)
        except:
            self.fail("OTree flatten failed unexpectedly.")
        try:
            _ = jax.tree.unflatten(treedef, vals)
        except:
            self.fail("OTree unflatten failed unexpectedly.")

    def test_update(self):
        class MyOTree(ojax.OTree):
            a: jax.Array = ojax.aux()
            b: float = ojax.child()
            c: typing.Any = ojax.ignore()
            e: dataclasses.InitVar[str]
            d: typing.ClassVar[typing.Any] = None
            f: io.BytesIO = ojax.alien(default_factory=io.BytesIO)

            def __init__(self, a, b, c, e) -> None:
                assert e == 'lol'
                self.assign_(a=a, b=b, c=c)

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
        try:
            my_otree = my_otree.update(
                f=None
            )
        except:
            self.fail("OTree update failed unexpectedly.")
        self.assertEqual(my_otree.f, None)
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

            def __init__(self, a, b, c, e) -> None:
                assert e == 'lol'
                self.assign_(a=a, b=b, c=c)

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

            def __init__(self, shift: float):
                self.assign_(shift=shift)

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
