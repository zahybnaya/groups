import unittest
import groups as gp
import networkx as nx
class TestGroups(unittest.TestCase):
    def setUp(self):
        self.seq = range(10)

    def readGraph(self,graphFileName):
        lines = file(graphFileName).readlines()
        edges = [edge[:-1].split(",")
                 for edge in lines if not edge.startswith('#')]
        G = nx.graph.Graph(edges)
        return G

    def test_updateInformation(self):
        # test the test
        G = self.readGraph("./ning_data/NingGroups.csv")
        v = 'melon\r'
        visibleG = G.subgraph([])
        CLOSED = set([])
        OPEN = gp.RND(visibleG,())
        groupsFound = set([])
        gp.updateInformation(G,v, visibleG, groupsFound, CLOSED, OPEN)
        #assert that discovery is set
        self.assertTrue(visibleG.node['money']['discovery']==8,"should be 1:"+str(visibleG.node['money']['discovery']))
        #assert that membership is set
        self.assertTrue(visibleG.node['money']['membership']==7,"should be 0:"+str(visibleG.node['money']['membership']))
        #assert that flips is set
        self.assertTrue(visibleG.node['money']['flips']==1,"should be 1:"+str(visibleG.node['money']['flips']))
        #assert that the groups are known

if __name__ == '__main__':
    unittest.main()
