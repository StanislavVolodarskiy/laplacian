import types

from laplacian import utils

class TestUtils(object):
    def test_pairs(self):
        assert isinstance(utils.pairs([1, 2]), types.GeneratorType)
        assert list(utils.pairs([])) == []
        assert list(utils.pairs([1])) == [(1, 1)]
        assert list(utils.pairs([1, 2])) == [(1, 2), (2, 1)]
        assert list(utils.pairs([1, 2, 3])) == [(1, 2), (2, 3), (3, 1)]
        assert list(utils.pairs([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4), (4, 1)]

    def test_pairs(self):
        assert isinstance(utils.triples([1, 2]), types.GeneratorType)
        assert list(utils.triples([])) == []
        assert list(utils.triples([1])) == [(1, 1, 1)]
        assert list(utils.triples([1, 2])) == [(1, 2, 1), (2, 1, 2)]
        assert list(utils.triples([1, 2, 3])) == [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
        assert list(utils.triples([1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4), (3, 4, 1), (4, 1, 2)]
        assert list(utils.triples([1, 2, 3, 4, 5])) == [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)]
