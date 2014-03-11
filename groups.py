from math import sqrt, log
from socket import gethostname
import ConfigParser as pyini
from itertools import chain, combinations
import csv
from FileLock import register
import networkx as nx
import operator
import os
import random
import sys
import time

do_lock = False
lockFile = '.locks'
examine_arms = True
acquired_nolead_connection = True
_isLog = {'UCB': False, 'INSTANCES': False, 'GRAPH': False, '*': False}


def discover_neighbors(G, visibleG, v, CLOSED):
    neighbors = G.neighbors(v)
    discovered = set(neighbors)
    discovered.difference_update(CLOSED)
    discovered.difference_update(set(visibleG.nodes()))
    visibleG.add_nodes_from(discovered, isLead=None)
    edges = G.edges([v])
    visibleG.add_edges_from(edges)
    return discovered


def isLogEnabled(name='*'):
    return _isLog.get(name, False)


def logger(name, msg):
    if _isLog.get(name, False):
        print msg


def has_option(ini, section, option):
    if ini.has_option(section, option):
        return True
    elif ini.has_option(pyini.DEFAULTSECT, option):
        return True
    elif ini.defaults().has_key(option):
        return True
    else:
        return False


def prompt_for_option(ini, section, option, prompt):
    if ini.has_option(section, option):
        value = eval(ini.get(section, option))
    elif ini.has_option(pyini.DEFAULTSECT, option):
        value = eval(ini.get(pyini.DEFAULTSECT, option))
    elif ini.defaults().has_key(option):
        value = eval(ini.defaults()[option])
    else:
        value = input(prompt)
    return value


def prompt_for_raw_option(ini, section, option, prompt):
    if ini.has_option(section, option):
        value = ini.get(section, option)
    elif ini.has_option(pyini.DEFAULTSECT, option):
        value = ini.get(pyini.DEFAULTSECT, option)
    elif ini.defaults().has_key(option):
        value = ini.defaults()[option]
    else:
        value = raw_input(prompt)
    return value


def get_option(ini, section, option, default):
    if ini.has_option(section, option):
        value = eval(ini.get(section, option))
    elif ini.has_option(pyini.DEFAULTSECT, option):
        value = eval(ini.get(pyini.DEFAULTSECT, option))
    elif ini.defaults().has_key(option):
        value = eval(ini.defaults()[option])
    else:
        value = default
    return value


def prompt_for_arguments(f):
    argnames = [f.func_code.co_varnames[i]
                for i in range(f.func_code.co_argcount)]
    required_argcount = f.func_code.co_argcount - len(f.func_defaults)
    argvalues = []
    for i in range(f.func_code.co_argcount):
        if i < required_argcount:
            argvalues.append(input("please enter \"" + argnames[i] + "\">"))
        else:
            value = raw_input(
                "please enter \"" + argnames[i] + "\" (default value:" + repr(f.func_defaults[i-required_argcount]) + ")>")
            if value == "":
                value = f.func_defaults[i-required_argcount]
            else:
                value = eval(value)
                argvalues.append(value)
                return argvalues


def prompt_for_argument_lists(f, ini, section):
    argnames = [f.func_name + "." + f.func_code.co_varnames[i]
                for i in range(f.func_code.co_argcount)]
    required_argcount = f.func_code.co_argcount - len(f.func_defaults)
    argvalues = []
    docdisplayed = False
    for i in range(f.func_code.co_argcount):
        if i < required_argcount:
            if (not docdisplayed) and (not has_option(ini, section, argnames[i])):
                docdisplayed = True
                print(f.__doc__)
                argvalues.append(prompt_for_option(
                    ini, section, argnames[i], "please list the values of \"" + argnames[i] + ":"))
            else:
                if docdisplayed:
                    value = prompt_for_raw_option(ini, section, argnames[
                                                  i], "please list the values of \"" + argnames[i] + "\" (default [" + str(f.func_defaults[i-required_argcount]) + "]):")
                    if value == "":
                        value = [f.func_defaults[i-required_argcount]]
                    else:
                        value = eval(value)
                        argvalues.append(value)
                else:
                    argvalues.append(
                        get_option(ini, section, argnames[i], [f.func_defaults[i-required_argcount]]))
    return argvalues


def stored_graphs(condition):
    graphFiles = os.listdir(graph_folder)
    graphFiles = filter(condition, graphFiles)
    for f in graphFiles:
        lines = file(graph_folder+f).readlines()
        if len(lines) == 0:
            continue
        edges = [edge[:-1].split(",")
                 for edge in lines if not edge.startswith('#')]
        G = nx.graph.Graph(edges)
        yield G
        pass


class FIFO():

    def __init__(self, vG, initialLeads):
        self._G = vG
        self._open = set([])
        self._queue = {}
        self._i = 0
        self._initialLeads = list(initialLeads)
        self.addall(initialLeads, None, None)

    def __len__(self):
        return len(self._open)

    def __iter__(self):
        return self

    def _get_initial_lead(self):
        ''' If there is an initial lead that hasn't been returned yet - return it'''
        if len(self._initialLeads) > 0:
            v = self._initialLeads.pop()
            self._open.remove(v)
            k_list = [k for (k, value) in self._queue.items() if v in value]
            k = k_list[0]
            self._queue[k].remove(v)
            if len(self._queue[k]) == 0:
                del self._queue[k]
                return v, k
            else:
                return None

    def next(self):
        # Initial leads must be expanded first
        initial_lead = self._get_initial_lead()
        if initial_lead is not None:
            return initial_lead
        if len(self._open) == 0:
            raise StopIteration()
        else:
            k = max(self._queue)
            while not len(self._queue[k]):
                del self._queue[k]
                k = max(self._queue)
            nbunch = self._queue[k]
            i = random.randint(0, len(nbunch)-1)
            v = nbunch[i]
            del nbunch[i]
            self._open.remove(v)
            if len(self._queue[k]) == 0:
                del self._queue[k]
            return v, k

    def addall(self, discovered, v, v_is_lead):
        discovered = set(discovered)
        discovered.difference_update(self._open)
        self._i -= 1
        self._queue[self._i] = list(discovered)
        self._open.update(discovered)


class KD(FIFO):

    def __init__(self, vG, initialLeads):
        FIFO.__init__(self, vG, initialLeads)

    def addall(self, discovered, v, v_is_lead):
        self._open.update(discovered)
        self.requeue()

    def requeue(self):
        self._queue.clear()
        ranks = self.getRanks()
        for v, k in ranks.items():
            nbunch = self._queue.get(k, [])
            nbunch.append(v)
            self._queue[k] = nbunch

    def getRanks(self):
        ranks = nx.degree(self._G, self._open)
        return ranks
class BysP(KD):
    pass

class RND(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        ranks = dict([(v, random.random()) for v in self._open])
        return ranks


class CC(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def addall(self, discovered, v, v_is_lead):
        if len(discovered) > 0:
            KD.addall(self, discovered, v, v_is_lead)

    def getRanks(self):
        ranks = nx.clustering(self._G, self._open)
        return ranks


class EC(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        ranks = {}
        for v, k in nx.eigenvector_centrality(self._G, max_iter=100, tol=1.0).items():
            if v in self._open:
                ranks[v] = k
                return ranks


class GROUPS(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G
        ranks = {}
        openList = self._open.copy()
        for v in openList:
            ranks[v] = G.node[v]["groups"]
        return ranks



class eGreedyBysP(BysP, RND):

    """ each epsilon - explore. Otherwise take best"""

    def __init__(self, G, initialLeads):
        BysP.__init__(self, G, initialLeads)

        def getRanks(self):
            epsi = self.calculateEpsilon()
            if random.random() < epsi:
                return RND.getRanks(self)
            else:
                return BysP.getRanks(self)

    def calculateEpsilon(self):
        c = self._getC()
        d = self._getD()
        pulls = len(
            set([n for n in self._G.nodes() if self._G.node[n]["isLead"] is not None]))
        K = len(
            set([v for v in self._G.nodes() if self._G.node[v]["isLead"] == True]))
        return (c*K)/(d*pulls)

    def _getC(self):
        return 1.0

    def _getD(self):
        return 1.0


class eGreedyKD(eGreedyBysP):

    """ For KD algorithm """

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

        def getRanks(self):
            epsi = self.calculateEpsilon()
            if random.random() < epsi:
                return RND.getRanks(self)
            else:
                return KD.getRanks(self)


class eGreedyKD001(eGreedyKD):

    def calculateEpsilon(self):
        return 0.01


class eGreedyKD01(eGreedyKD):

    def calculateEpsilon(self):
        return 0.1


class eGreedyKD005(eGreedyKD):

    def calculateEpsilon(self):
        return 0.05


class eGreedylogKDT(eGreedyKD):

    """logT for eGreedy"""

    def __init__(self, G, initalLeads):
        eGreedyKD.__init__(self, G, initalLeads)

        def calculateEpsilon(self):
            pulls = len(
                set([n for n in self._G.nodes() if self._G.node[n]["isLead"] is not None]))
            return (log(pulls)/float(pulls))


class eGreedylogKDT_SQRT(eGreedyKD):

    """logT for eGreedy"""

    def __init__(self, G, initalLeads):
        eGreedyKD.__init__(self, G, initalLeads)

        def calculateEpsilon(self):
            pulls = len(
                set([n for n in self._G.nodes() if self._G.node[n]["isLead"] is not None]))
            return sqrt((log(pulls)/float(pulls)))


class eGreedylogT_SQRT(eGreedyBysP):

    """logT for eGreedy"""

    def __init__(self, G, initalLeads):
        eGreedyBysP.__init__(self, G, initalLeads)

        def calculateEpsilon(self):
            pulls = len(
                set([n for n in self._G.nodes() if self._G.node[n]["isLead"] is not None]))
            return sqrt((log(pulls)/float(pulls)))


class eGreedy10(eGreedyBysP):

    def _getD(self):
        return 10.0


class UCB(BysP):

    """ UCB - Looks at each lead as arm """

    def __init__(self, G, initialLeads):
        self.hits = {}
        self.initial_leads_copy = initialLeads[:]
        BysP.__init__(self, G, initialLeads)

    def getRankOfArm(self, ns):
        G = self._G
        return max([G.node[l]["promise"] + G.node[l]["exploration"] for l in ns]+[0.0])

    def updateHits(self, v, v_is_lead, leads):
        if v_is_lead:
            for l in leads:
                self.hits[l] = 1
        else:
            leads &= set(self._G.neighbors(v))
            for l in leads:
                self.hits[l] = self.hits.get(l, 0) + 1

    def addall(self, discovered, v, v_is_lead):
        logger('UCB', 'expanded ' + str(v) + ' is lead? ' + str(v_is_lead))
        leads = set(
            [x for x in self._G.nodes() if self._G.node[x]["isLead"] == True])
        if v is not None:
            if isLogEnabled('UCB'):
                log_leads = leads & set(self._G.neighbors(v))
                print ' potential of: ' + str(log_leads) + " all leads:"+str(leads)
                self.updateHits(v, v_is_lead, leads)
                BysP.addall(self, discovered, v, v_is_lead)

    def calculateExploitationValues(self, G, leads, notleads, potentials):
        logger('UCB', self.hits)
        for v in leads:
            NL = 0.0
            ns = set(G.neighbors(v))
            L = 1.0 * len(ns & leads)
            NL = 1.0 * len(ns & notleads)
            G.node[v]["promise"] = self._getPromiseFactor(L, NL)
            logger('UCB', v+':'+str(G.node[v]["promise"]))

    def calculateExplorationValues(self, G, leads, v, notleads):
        total_pulls = sum(self.hits.values())
        for v in leads:
            G.node[v]["exploration"] = self._getXplorationFactor(
                self.hits.get(v, 0), total_pulls)

    def _getXplorationFactor(self, hits, TTL):
        if hits == 0:
            return float('inf')
        return self._getCp()*sqrt(log(max(TTL, 1))/(hits))

    def _getCp(self):
        ns = self.initial_leads_copy
        cp = max([self._G.node[l]["promise"] for l in ns]+[0.4])
        return cp

    def getRanks(self):
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        self.calculateExploitationValues(G, leads, notleads, potentials)
        self.calculateExplorationValues(G, leads, v, notleads)
        if isLogEnabled('UCB'):
            for l in leads:
                print l,
                print 'promise:' + str(G.node[l].get("promise", 'unknown')),
                print 'exploration:' + str(G.node[l].get("exploration", 'unknown'))
                ranks = {}
                openList = self._open.copy()
                for v in openList:
                    ns = set(G.neighbors(v))
                    assert len(ns) == len(ns & leads)
                    P = self.getRankOfArm(ns)
                    ranks[v] = P
                    if isLogEnabled('UCB'):
                        max_rs = [r for r in ranks if ranks[r]
                                  == max(ranks.values())]
                        for r in max_rs:
                            print str(r) + ' friends of ' + str(set(G.neighbors(r)))
                            return ranks


class UCB_NORESET(UCB):

    def updateHits(self, v, v_is_lead, leads):
        leads &= set(self._G.neighbors(v))
        for l in leads:
            self.hits[l] = self.hits.get(l, 0) + 1


def print_lead_data(leads, G, name):
    print "************* " + name + "********************"
    for l in sorted(leads):
        print "p:"+str(G.node[l]["promise"]) + " x:" + str(G.node[l]["exploration"])
        print "********************************************"


class UCB_ADD(UCB):

    """ Looks at the sum of all neighbors (superarms) """

    def getRankOfArm(self, ns):
        G = self._G
        return sum([G.node[l]["promise"] + G.node[l]["exploration"] for l in ns])


class UCB_COM(UCB):

    """ Combinatorial version"""

    def __init__(self, G, initialLeads):
        UCB.__init__(self, G, initialLeads)

    def _getCp(self):
        cp = max(self.exploitation.values())
        return cp

    def getRankOfArm(self, G, ns):
        ns = tuple(sorted(ns))
        if ns in self.exploitation:
            return self.exploitation[ns] + self.exploration[ns]
        else:
            return 0.5

    def calculateExploitationValues(self, G, leads, notleads, potentials):
        openList = self._open.copy()
        nodes = G.nodes()
        self.exploration = {}
        self.exploitation = {}
        for v in openList:
            ns = set(G.neighbors(v))
            if not len(ns):
                continue
            assert len(ns) == len(ns & leads)
            ns = tuple(sorted(ns))
            if ns in self.exploitation:
                continue
            ns_of_group = set(nodes)
            for l in ns:
                l_ns = set(G.neighbors(l))
                l_L = 1.0*len(l_ns & leads)
                l_NL = 1.0*len(l_ns & notleads)
                G.node[l]["promise"] = self._getPromiseFactor(l_L, l_NL)
                ns_of_group = l_ns & ns_of_group
                L = 1.0*len(ns_of_group & leads)
                NL = 1.0*len(ns_of_group & notleads)
                self.exploitation[ns] = self._getPromiseFactor(L, NL)
                self.exploration[ns] = L+NL

    def calculateExplorationValues(self, G, leads, v, notleads):
        # total_pulls=sum(self.exploration.values())
        nodes = G.nodes()
        total_pulls = len(
            set([v for v in nodes if G.node[v]["isLead"] is not None]))
        for v in self.exploration:
            self.exploration[v] = self._getXplorationFactor(
                self.exploration[v], total_pulls)


class UCB_COM_RESET(UCB):

    """ The COM version that gurantees regret"""

    def __init__(self, G, initialLeads):
        self.arm_hits = {}
        self.total_pulls = 0
        UCB.__init__(self, G, initialLeads)

    def _getCp(self):
        cp = 1.0  # 0.5#max(self.exploitation.values()+[0])
        return cp

    def addall(self, discovered, v, v_is_lead):
        UCB.addall(self, discovered, v, v_is_lead)
        if v is None:
            return
        ns = set(G.neighbors(v))
        ns &= set(
            [v for v in self._G.nodes() if self._G.node[v]["isLead"] == True])
        if not len(ns):
            return
        ns = tuple(sorted(ns))
        if ns not in self.arm_hits:
            self.arm_hits[ns] = [self.total_pulls, 0]
            self.arm_hits[ns][1] += 1
            self.total_pulls += 1

    def getRankOfArm(self, G, ns):
        ns = tuple(sorted(ns))
        P = 1.0 - reduce(operator.mul, [1.0 - G.node[l]["promise"]
                         for l in ns], 1)
        if ns in self.arm_hits:
            return P + len(ns)*self.exploration[ns]
        else:
            return P

    def calculateExploitationValues(self, G, leads, notleads, potentials):
        self._leads = leads
        for l in leads:
            ns = set(G.neighbors(l))
            if not len(ns):
                continue
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            G.node[l]["promise"] = self._getPromiseFactor(L, NL)
            # Detect new arms
            #        openList=self._open.copy()
            #        for v in openList:
            #            ns = tuple(sorted(set(G.neighbors(v))))
            #            if not len(ns): continue
            #            if ns not in self.arm_hits:
            #                self.arm_hits[ns] = [self.total_pulls,1]

#    def calculateExploitationValues(self, G, leads, notleads, potentials):
#        openList=self._open.copy()
#        nodes = G.nodes()
#        self.exploitation={}
#        self.exploration={}
#        self._leads = leads
#        for v in openList:
#            ns = set(G.neighbors(v))
#            if not len(ns): continue
#            assert len(ns) == len(ns & leads)
#            ns=tuple(sorted(ns))
#            if ns in self.exploitation: continue
#            ns_of_group=set(nodes)
#            for l in ns:
#                l_ns = set(G.neighbors(l))
#                l_L  = 1.0*len(l_ns & leads)
#                l_NL = 1.0*len(l_ns & notleads)
#                G.node[l]["promise"] = self._getPromiseFactor(l_L, l_NL)
#                ns_of_group = l_ns & ns_of_group
#            L  = 1.0*len(ns_of_group & leads)
#            NL = 1.0*len(ns_of_group & notleads)
#            self.exploitation[ns] = self._getPromiseFactor(L, NL)
# Detect new arms
#            if ns not in self.hits:
#                self.arm_hits[ns] = [self.total_pulls,1]

    def calculateExplorationValues(self, G, leads, v, notleads):
        self.exploration = {}
        for v in self.arm_hits:
            self.exploration[v] = self._getXplorationFactor(
                self.arm_hits[v][1], self.total_pulls-self.arm_hits[v][0])


def arms_stat_begin():
    #fieldnames = "arm", "creationTime","hits", "potentials", "leads","non_leads"
    fieldnames = "hits", "opt_len", "opt_potentials", "opt_leads", "opt_noleads", "opt_exploitation", "opt_mu", "opt_bysp", "opt_exploration", "sel_len", "sel_potentials", "sel_leads", "sel_noleads", "sel_exploitation", "sel_mu", "sel_bysp", "sel_exploration", "num_arms", "num_dead", "num_new", "is_opt", "sel_hits", "leads", "notleads"
    armf = csv.DictWriter(file("arms_%s.csv", "w"),
                          fieldnames, "", lineterminator="\n")
    armf.writerow(dict(zip(fieldnames, fieldnames)))
    return armf


def neighborsOfArm(G, arm):
    if len(arm) == 0:
        return set([])
    ns = set(G.nodes())
    for l in arm:
        ns &= set(G.neighbors(l))
        otherLeads = set([ol for ol in leadsFound if ol not in arm])
        ns = set([n for n in ns if len(set(G.neighbors(n)) & otherLeads) == 0])
        return ns


class SN_UCB(UCB):

    """ Takes the average utility """

    def __init__(self, G, initialLeads):
        if examine_arms:
            self.armf = arms_stat_begin()
            self.arm_hits = {}
            UCB.__init__(self, G, initialLeads)

    def _getCp(self):
        cp = 1.0*max(self.exploitation.values()+[0])
        return cp

    def createArmsIfNecessary(self, v_is_lead, leads):
        if v_is_lead:
            openList = self._open.copy()
            for vol in openList:
                arm = set(self._G.neighbors(vol)) & leads
                arm = tuple(sorted(arm))
                if arm not in self.arm_hits:
                    self.arm_hits[arm] = [-1, 0]

    def realPromise(self, arm, justExpanded, target):
        if len(set([_ for _ in neighborsOfArm(self._G, arm) if self._G.node[_]["isLead"] == None])) == 0 and justExpanded not in neighborsOfArm(self._G, arm):
            return -1
        if len(arm) == 0:
            return -1
        a_neigbors = neighborsOfArm(G, arm)
        L = 0.0
        NL = 0.0
        for p in a_neigbors:
            if isLead(G, target, p):
                L += 1.0
            else:
                NL += 1.0
                return self._getPromiseFactor(L, NL)

    def findOptimalArm(self, justExpanded):
        """ The arm with the best exploitation in G"""
        if len(self.arm_hits) == 0:
            maxArm = ()
        else:
            maxArm = max(self.arm_hits, key=lambda x:
                         self.realPromise(x, justExpanded, targetID))
            optimalNeigbors = neighborsOfArm(self._G, maxArm)
            return maxArm, optimalNeigbors, self.realPromise(maxArm, justExpanded, targetID)

    def writeArmStats(self, selected, v):
        leads = set(
            [_ for _ in self._G.nodes() if self._G.node[_]["isLead"] == True])
        noleads = set(
            [_ for _ in self._G.nodes() if self._G.node[_]["isLead"] == False])
        potentials = set(
            [_ for _ in self._G.nodes() if self._G.node[_]["isLead"] == None])
        optimalArm, optimalNeigbors, real_optimal = self.findOptimalArm(v)
        dead_arms = [_ for _ in self.arm_hits if len(
            neighborsOfArm(self._G, _) & potentials) == 0]
        new_arms = [_ for _ in self.arm_hits if neighborsOfArm(
            self._G, _) <= potentials and _ not in dead_arms]
        nsNeigbors = neighborsOfArm(self._G, selected)
        # We had one pull (We can detect only one lead/nolead)
        if len(set(nsNeigbors) & (leads | noleads)) == 1:
            new_arms.append(selected)
            real_selected = self.realPromise(selected, v, targetID)
            selected_bysp = 1.0 - \
                reduce(operator.mul, [1.0 - self._getPromiseFactor(1.0*len(set(G.neighbors(l)) & leads), 1.0*len(set(G.neighbors(l)) & noleads))
                       for l in selected], 1.0)
            optimal_bysp = 1.0 - \
                reduce(operator.mul, [1.0 - self._getPromiseFactor(1.0*len(set(G.neighbors(l)) & leads), 1.0*len(set(G.neighbors(l)) & noleads))
                       for l in optimalArm], 1.0)
            d = {"hits": len(leads) + len(noleads),
                 "opt_len": len(optimalArm),
                 "opt_potentials": len(optimalNeigbors & potentials),
                 "opt_leads": len(optimalNeigbors & leads),
                 "opt_noleads": len(optimalNeigbors & noleads),
                 "opt_exploitation": self.exploitation.get(optimalArm, '?'),
                 "opt_mu": real_optimal,
                 "opt_bysp": optimal_bysp,
                 "opt_exploration": self.exploration.get(optimalArm, '?'),
                 "sel_len": len(selected),
                 "sel_potentials": len(nsNeigbors & potentials),
                 "sel_leads": len(nsNeigbors & leads),
                 "sel_noleads": len(nsNeigbors & noleads),
                 "sel_exploitation": self.exploitation.get(selected, '?'),
                 "sel_mu": real_selected,
                 "sel_bysp": selected_bysp,
                 "sel_exploration": self.exploration.get(selected, '?'),
                 "num_arms": len(self.arm_hits),
                 "num_dead": len(dead_arms),
                 "num_new": len(new_arms),
                 "is_opt": real_optimal == real_selected,
                 "sel_hits": self.arm_hits.get(selected, ['?', '?'])[1],
                 "leads": len(leads),
                 "notleads": len(noleads)
                 }
            self.armf.writerow(d)

    def addall(self, discovered, v, v_is_lead):
        if examine_arms and v is not None:
            leads = set(
                [vl for vl in self._G.nodes() if self._G.node[vl]["isLead"] == True])
            ns = set(self._G.neighbors(v)) & leads
            ns = tuple(sorted(ns))
            if len(ns):
                self.writeArmStats(ns, v)
                UCB.addall(self, discovered, v, v_is_lead)

    def getRankOfArm(self, ns):
        ns = tuple(sorted(ns))
        if not len(ns):
            return
        leads = set(
            [vl for vl in self._G.nodes() if self._G.node[vl]["isLead"] == True])
        notleads = set(
            [vl for vl in self._G.nodes() if self._G.node[vl]["isLead"] == False])
        # *float(len(ns))/max([len(arm) for arm in self.arm_hits]+[1.0])
        P = self.exploitation[ns]
        if ns in self.arm_hits:
            return P + self.exploration[ns]
        else:
            return P

    def allArmCombinations(self, leads=None):
        if leads is None:
            leads = set(
                [v for v in self._G.nodes() if self._G.node[v]["isLead"] == True])
            return chain(*[combinations(leads, r) for r in range(1, len(leads)+1)])

    def calculateExploitationValues(self, G, leads, notleads, potentials):
        self._leads = leads
        self.exploitation = {}
        totalPulls = len(leads) + len(notleads)
        openList = self._open.copy()
        unknown_arms = set([])
        VIRTUAL = 25.0
        for v in openList:
            arm = set(self._G.neighbors(v)) & leads
            arm = tuple(sorted(arm))
            if arm in self.exploitation or len(arm) == 0:
                continue
            ns = neighborsOfArm(G, arm)
            if not len(ns):
                continue
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            if L+NL == 0 and len(arm) > 2 and len(leadsFound) > 0:
                unknown_arms.add(arm)
                continue
            P = 1.0 - \
                reduce(
                    operator.mul, [1.0 - self._getPromiseFactor(
                        1.0*len(set(G.neighbors(l)) & leads),
                        1.0*len(set(G.neighbors(l)) & notleads)) for l in arm], 1.0)
            if arm not in self.arm_hits:
                self.arm_hits[arm] = [totalPulls, L+NL+VIRTUAL]
            else:
                self.arm_hits[arm][1] = L+NL+VIRTUAL
                self.exploitation[arm] = (
                    self._getPromiseFactor(L, NL) * (L+NL) + VIRTUAL * P) / (VIRTUAL+L+NL)
                # handle unknown arms
                for n_arm in unknown_arms:
                    newPotentials = neighborsOfArm(G, n_arm)
                    lastLeadFound = leadsFound[-1]
                    possible_arm = set(n_arm)
                    if lastLeadFound in possible_arm:
                        possible_arm.remove(lastLeadFound)
                        possible_arm = tuple(sorted(possible_arm))
                        howmuch = 0.0
                        possible_arm_score = 0.0
                        if possible_arm in self.exploitation.keys():
                            friendsOf = reduce(
                                lambda s1, s2: s1 & s2, [set(G.neighbors(l)) for l in possible_arm])
                            howmuch = len(newPotentials & friendsOf)
                            possible_arm_score = self.exploitation[
                                possible_arm]
                            P = 1.0 - \
                                reduce(operator.mul, [1.0 - self._getPromiseFactor(1.0*len(set(G.neighbors(l)) & leads), 1.0*len(set(G.neighbors(l)) & notleads))
                                       for l in n_arm], 1.0)
                            self.exploitation[n_arm] = (
                                howmuch/len(newPotentials))*possible_arm_score + (1-(howmuch/len(newPotentials))) * P
                            if n_arm not in self.arm_hits:
                                self.arm_hits[n_arm] = [
                                    totalPulls+VIRTUAL, VIRTUAL]
                            else:
                                self.arm_hits[n_arm][1] = VIRTUAL

    def calculateExplorationValues(self, G, leads, v, notleads):
        self.exploration = {}
        total_pulls = len(leads) + len(notleads)
        for v in self.arm_hits:
            hits = max(self.arm_hits[v][1], 1)
            self.exploration[v] = self._getXplorationFactor(
                hits, max(total_pulls-self.arm_hits[v][0], hits))


class OPT_ARM(SN_UCB):

    """ Pulls the best arm at every time"""

    def getRankOfArm(self, ns):
        return self.realPromise(ns, None, targetID)

    def calculateExplorationValues(self, G, leads, v, notleads):
        if examine_arms:
            SN_UCB.calculateExplorationValues(self, G, leads, v, notleads)

            def calculateExploitationValues(self, G, leads, notleads, potentials):
                if examine_arms:
                    SN_UCB.calculateExploitationValues(
                        self, G, leads, notleads, potentials)


class SN_UCB_TIEBREAK_PLEADS(SN_UCB):

    """ Tie breaking according BysP  """

    def getRanks(self):
        ranks = SN_UCB.getRanks(self)
        if None in ranks.values() or len(ranks) == 0:
            return ranks
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentialleads = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            ns = set(G.neighbors(v))
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            G.node[v]["promise"] = self._getPromiseFactor(L, NL)
            maxRank = max(ranks.values())
            maxItems = [r for r in ranks if ranks[r] == maxRank]
            if maxRank == float('inf'):
                maxRank = max(
                    [r for r in ranks.values() if r != maxRank]+[0.0])
                for mi in maxItems:
                    ns = set(G.neighbors(mi))
                    pleads = neighborsOfArm(G, ns) & potentialleads
                    ranks[mi] = maxRank+(potentialleads - len(pleads))
                    return ranks


class SN_UCB_TIEBREAK_KD(SN_UCB):

    """ Tie breaking according BysP  """

    def getRanks(self):
        ranks = SN_UCB.getRanks(self)
        if None in ranks.values() or len(ranks) == 0:
            return ranks
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        for v in leads:
            ns = set(G.neighbors(v))
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            G.node[v]["promise"] = self._getPromiseFactor(L, NL)
            maxRank = max(ranks.values())
            maxItems = [r for r in ranks if ranks[r] == maxRank]
            if maxRank == float('inf'):
                maxRank = max(
                    [r for r in ranks.values() if r != maxRank]+[0.0])
                for mi in maxItems:
                    ns = set(G.neighbors(mi))
                    #P = 1.0 - reduce(operator.mul,[1.0 - G.node[l]["promise"] for l in ns],1)
                    ranks[mi] = maxRank+len(ns)
                    return ranks


class UCB_COM_BysP(UCB_COM):

    def _getCp(self):
        return UCB_COM._getCp(self)*5.0

    def getRankOfArm(self, G, ns):
        P = 1.0 - reduce(operator.mul, [1.0 - G.node[l]["promise"]
                         for l in ns], 1)
        return P


class UCB_COM5(UCB_COM):

    def _getCp(self):
        return UCB_COM._getCp(self)*5.0


class UCB_COM05(UCB_COM):

    def _getCp(self):
        return UCB_COM._getCp(self)*0.5


class UCT(UCB):

    """ A tree-based UCB. It uses also an epsilon-greedy to decide on when to choose internal leafs"""

    def __init__(self, G, initialLeads):
        self.epsilon = 0.15
        self.leadTree = {'root': {'r': 0, 'v': 0.5, 'S': []}}
        self.leads_trail = []
        UCB.__init__(self, G, initialLeads)

    def updateTrail(self, v, isLead):
        """ trail=list of leads from root to end"""
        if isLead:
            self.leadTree[self.leads_trail[-1]]['S'].append(v)
            self.leadTree[v] = {'r': 0, 'v': 0.5, 'S': []}
            for l in self.leads_trail:
                self.leadTree[l]['v'] = (
                    self.leadTree[l]['v']*self.leadTree[l]['r'] + (0.0, 1.0)[isLead]) / (self.leadTree[l]['r']+1)
                self.leadTree[l]['r'] += 1

    def addall(self, discovered, v, v_is_lead):
        self.updateTrail(v, v_is_lead)
        UCB.addall(self, discovered, v, v_is_lead)

    def getRanks(self):
        G = self._G
        self.leads_trail = self.selectRoute()
        openList = self._open.copy()
        endLead = self.leads_trail[-1]
        if endLead == 'root':
            chosen = []
        else:
            chosen = set(G.neighbors(endLead)) & set(openList)
            ranks = {}
            byspRanks = BysP.getRanks(self)
            for v in openList:
                ranks[v] = (9999.0)*(-1.0, 1.0)[v in chosen] + byspRanks[v]
                return ranks

    def uctMax(self, successors, r):
        return max(successors, key=lambda s: self._getXplorationFactor(self.leadTree[s]['r'], r) + self.leadTree[s]['v'])

    def _getCp(self):
        return self.leadTree['root']['v']

    def selectRoute(self):
        leads_trail = []
        best = 'root'
        leads_trail.append(best)
        while True:
            if len(self._initialLeads):
                break
            if random.random() < self.epsilon and best != 'root':
                break
            successors = self.leadTree[best]['S']
            if not len(successors):
                break
            r = self.leadTree[best]['r']
            best = self.uctMax(successors, r)
            leads_trail.append(best)
            return leads_trail


class AvgP(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            NL = 0.0
            NA = 0.0
            ns = set(G.neighbors(v))
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            NA = 1.0*len(ns & potentials)
            G.node[v]["promise"] = self._getPromiseFactor(L, NL, NA)
        ranks = {}
        openList = self._open.copy()
        for v in openList:
            ns = set(G.neighbors(v))
            # assert all neighbors are leads
            assert len(ns) == len(ns & leads)
            #P = 1.0 - reduce(operator.mul,[1.0 - G.node[l]["promise"] for l in ns],1)
            if(len(ns) > 0):
                P = (sum(G.node[l]["promise"] for l in ns))/(len(ns))
            else:
                P = 0
                G.node[v]["promising"] = P
                ranks[v] = P
                return ranks

    def _getPromiseFactor(self, L, NL, NA):
        if L+NL > 0:
            return L / (L+NL)
        else:
            return 0.5


class MaxP(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G

        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            NL = 0.0
            NA = 0.0
            ns = set(G.neighbors(v))
            L = 1.0*len(ns & leads)
            NL = 1.0*len(ns & notleads)
            NA = 1.0*len(ns & potentials)
            G.node[v]["promise"] = self._getPromiseFactor(L, NL, NA)

        ranks = {}
        openList = self._open.copy()
        for v in openList:
            ns = set(G.neighbors(v))
            # assert all neighbors are leads
            assert len(ns) == len(ns & leads)
            #P = 1.0 - reduce(operator.mul,[1.0 - G.node[l]["promise"] for l in ns],1)
            if(len(ns) > 0):
                P = max(G.node[l]["promise"] for l in ns)
            else:
                P = 0
                G.node[v]["promising"] = P
                ranks[v] = P
                return ranks

    def _getPromiseFactor(self, L, NL, NA):
        if L+NL > 0:
            return L / (L+NL)
        else:
            return 0.5


class SMOOTHED_PROMISING(BysP):

    def __init__(self, G, initialLeads):
        PROMISING.__init__(self, G, initialLeads)

    def _getPromiseFactor(self, L, NL, NA):
        if L+NL > 0:
            return (L+1) / (L+NL+1)
        else:
            return 0.5


class PL(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            ns = set(G.neighbors(v))
            L = 1.0*len(ns & leads)
            if(len(ns) == 0):
                G.node[v]["LConnected"] = 0
            else:
                G.node[v]["LConnected"] = float(L)/float(len(ns))
                ranks = {}
                openList = self._open.copy()
                for v in openList:
                    ns = set(G.neighbors(v))
                    # assert all neighbors are leads
                    assert len(ns) == len(ns & leads)
                    PL = sum(G.node[l]["LConnected"] for l in ns)
                    if(len(ns) == 0):
                        ranks[v] = 0
                    else:
                        ranks[v] = float(PL) / float(len(ns))
                        return ranks


class PNL(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            NL = 0.0
            ns = set(G.neighbors(v))
            NL = 1.0*len(ns & notleads)
            if(len(ns) == 0):
                G.node[v]["NLConnected"] = 0
            else:
                G.node[v]["NLConnected"] = float(NL)/float(len(ns))
                ranks = {}
                openList = self._open.copy()
                for v in openList:
                    ns = set(G.neighbors(v))
                    # assert all neighbors are leads
                    assert len(ns) == len(ns & leads)
                    PNL = sum(G.node[l]["NLConnected"] for l in ns)
                    if(len(ns) == 0):
                        ranks[v] = 0
                    else:
                        ranks[v] = float(PNL) / float(len(ns))
                        return ranks


class PP(KD):

    def __init__(self, G, initialLeads):
        KD.__init__(self, G, initialLeads)

    def getRanks(self):
        G = self._G
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        potentials = set([v for v in nodes if G.node[v]["isLead"] == None])
        for v in leads:
            NA = 0.0
            ns = set(G.neighbors(v))
            NA = 1.0*len(ns & potentials)
            if(len(ns) == 0):
                G.node[v]["PPConnected"] = 0
            else:
                G.node[v]["PPConnected"] = float(NA)/float(len(ns))
                ranks = {}
                openList = self._open.copy()
                for v in openList:
                    ns = set(G.neighbors(v))
                    # assert all neighbors are leads
                    assert len(ns) == len(ns & leads)
                    PP = sum(G.node[l]["PPConnected"] for l in ns)
                    if(len(ns) == 0):
                        ranks[v] = 0
                    else:
                        ranks[v] = float(PP) / float(len(ns))
                        return ranks


def getGroups(G, p):
    """Find the groups for profile p"""
    return G.neighbors(p)


def updateInformationList(initialGraph, visibleG, groupsFound):
    """ updates the information of vertices in initialGraph """
    for v in initialGraph:
        updateInformation(v, visibleG, groupsFound)

def updateGroupStats(v, visibleG, G):
    "Update the number of groups"
    visibleG.node[v]["groups"] = len(set(G.neighbors(v)))

def updateInformation(v, visibleG, groupsFound):
    """ updates the information by pulling profile v"""
    visibleG.add_node(v)
    newGroups = discover_neighbors(G, visibleG, v, CLOSED)
    groupsFound.update(newGroups)
    updateGroupStats(v,visibleG,G)
    if newGroups is None:
        return
    newProfiles = [
        p for g in newGroups for p in discover_neighbors(G, visibleG, g, CLOSED)]
    OPEN.addall(newProfiles, v, True)


loggingDeltaTime = 1


def createPotentialInstances(potential, visibleGraph, reqestCount):
    res = {}
    for a in algo:
        tmp = a(visibleGraph, potential)
        algoRank = tmp.getRanks()
        for id in algoRank:
            if id in res:
                res[id] = res[id]+[algoRank[id]]
            else:
                res[id] = [algoRank[id]]
                for id in algoRank:
                    res[id] = [reqestCount]+res[id]
                    return res


def logInstances(target, OPEN, G, instancesSet):
    with open(instancesLogDir+'\\'+target+'_'+OPEN.__class__.__name__+'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        nodes = G.nodes()
        leads = set([v for v in nodes if G.node[v]["isLead"] == True])
        notleads = set([v for v in nodes if G.node[v]["isLead"] == False])
        for cycle in instancesSet:
            for instanceId in instancesSet[cycle]:
                # post-process the results
                assert (instanceId in leads) or (instanceId in notleads)
                if instanceId in leads:
                    instancesSet[cycle][instanceId] = instancesSet[
                        cycle][instanceId]+['1']
                else:
                    instancesSet[cycle][instanceId] = instancesSet[
                        cycle][instanceId]+['0']
                    # write the results to file
                    writer.writerow(instancesSet[cycle][instanceId])

#
#    Graph Logging
#
graphLoggingHeader = [
    "#Iteration", "Lcount", "NLcount", "PLcount", "nodeCount", "edgeCount", "density",
    "numberC", "avgC", "giantC", "clusteringCoef", "CC_target", "Betweeness", "Lratio", "NLratio", "PLratio"]


def createGraphInstance(visibleG, reqestCount, targetID):
    stats = {}
    stats["#Iteration"] = reqestCount
    cG = visibleG.copy()
    cG.add_node(targetID)
    edges = []
    nodes = visibleG.nodes()
    leads = set([v for v in nodes if visibleG.node[v]["isLead"] == True])
    notLeads = set([v for v in nodes if visibleG.node[v]["isLead"] == False])
    potentialLeads = set(nodes) - leads - notLeads
    stats["Lcount"] = len(leads)
    stats["NLcount"] = len(notLeads)
    stats["PLcount"] = len(potentialLeads)

    stats["Lratio"] = float(len(leads))/len(nodes)
    stats["NLratio"] = float(len(notLeads))/len(nodes)
    stats["PLratio"] = float(len(potentialLeads))/len(nodes)

    n = cG.number_of_nodes()
    m = cG.number_of_edges()
    d = float(m)/float(n)

    stats["nodeCount"] = n
    stats["edgeCount"] = m
    stats["density"] = d

    comps = nx.connected_components(cG)
    comps = [len(comp) for comp in comps]

    stats["numberC"] = len(comps)
    stats["avgC"] = sum(comps)/len(comps)
    stats["giantC"] = max(comps)
    stats["clusteringCoef"] = nx.algorithms.cluster.average_clustering(cG)

    stats['CC_target'] = nx.clustering(cG, targetID)
    stats['Betweeness'] = nx.betweenness_centrality(cG)[targetID]
    return stats


def logGraph(targetID, OPEN, graphSet):
    with open(graphLogsDir+'\\'+targetID+'_'+OPEN.__class__.__name__+'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if(len(graphSet) == 0):
            writer.writerow('No valid reults')
            return
        else:
            # writer.writerow(graphLoggingHeader)
            iterations = graphSet.keys()
            for i in sorted(iterations, key=lambda(x): int(x)):
                iterationData = graphSet[i]
                output = []
                for h in graphLoggingHeader:
                    output += [iterationData[h]]
                    writer.writerow(output)


#
# START
# {id,reqestCount,PL,PNL,PP,AvgP,BysP,CC,EC,KD,MaxP,UCB}
ini = pyini.ConfigParser()
hasini = False
if len(sys.argv) >= 2:
    if len(sys.argv[1]) > 0:
        hasini = True
        ini.read(sys.argv[1])

if not hasini:
    inifile = os.path.basename(sys.argv[0])
    inifile += ".ini"
    if os.path.exists(inifile):
        ini.read(inifile)
        hasini = True
        sections = ini.sections()
        if len(sections) == 0:
            sections = [pyini.DEFAULTSECT]

print sections
for section in sections:
    print section
    resultfileprefix = get_option(
        ini, section, "output_file_prefix", str(section))
    graph_name_prefix = prompt_for_raw_option(
        ini, section, "graph_name_prefix", "Enter network name prefix (not a path):")
    graph_folder = prompt_for_raw_option(
        ini, section, "graph_folder", "Enter network folder path:")
    goal = prompt_for_option(ini, section, "goal", "Goal ]:")
    net_gen = stored_graphs(lambda x: x.endswith(".csv")
                            and (not x.startswith(resultfileprefix)))
    seed_counts = prompt_for_option(
        ini, section, "seed_counts", "List the seed counts [int [,int [...]]]:")
    domain = prompt_for_option(
        ini, section, "domain", "List the Domain[tonic/comm]:")
    heuristics = prompt_for_option(
        ini, section, "heuristics", "List the heuristics:")
    executions = prompt_for_option(
        ini, section, "executions", "List of executions:")
    initials = prompt_for_option(
        ini, section, "initialGroups", "initialGroups:")
    fieldnames = "source", "targetID", "n", "m", "density", "numberC", "avgC", "giantC", "clusteringCoef", "totalLeads", "alg", "initialSeeds", "open", "closed", "profileID", "heuristic", "lastIsLead", "IsLeadRequestsNum", "groupsFound", "notleadsFound", "time"
    resultf = csv.DictWriter(
        file(str(resultfileprefix) + str(gethostname())+"_%d.csv" %
             time.time(), "w"), fieldnames, "", lineterminator="\n")
    resultf.writeheader()
    stats = {}
    for seed_count in seed_counts:
        tcount = 0
        for G in net_gen:
            if do_lock:
                if register(lockFile, resultfileprefix+section+'\n'):
                    continue
            tcount += 1
            print tcount
            assert isinstance(G, nx.Graph)
            stats["initialSeeds"] = seed_count
            stats["source"] = section
            stats["targetID"] = 'NA'
            G_notarget = G.copy()
            n = G.number_of_nodes()
            m = G.number_of_edges()
            d = m/n
            stats["n"] = n
            stats["m"] = m
            stats["density"] = d
            comps = nx.connected_components(G_notarget)
            comps = [len(comp) for comp in comps]
            stats["numberC"] = len(comps)
            stats["avgC"] = sum(comps)/len(comps)
            stats["giantC"] = max(comps)
            stats[
                "clusteringCoef"] = nx.algorithms.cluster.average_clustering(G)
            stats["totalLeads"] = 'NA'
            stats["lastIsLead"] = 'NA'
            random.seed(0)
            initialGroups = initials
            for OPEN in heuristics:
                open_class = OPEN
                for execution in executions:
                    reqestCount = 0
                    T = time.time()
                    stats["alg"] = open_class.__name__+str(execution)
                    groupsFound = set([])
                    initialProfiles = [
                        gg for g in initialGroups for gg in G.neighbors(g)]
                    """Currently a single profile from each group"""
                    initialRepresentatives = [random.choice(G.neighbors(g))
                                              for g in initialGroups]
                    visibleG = G.subgraph(initialProfiles + initialGroups)
                    CLOSED = set([])
                    OPEN = open_class(visibleG, initialRepresentatives)
                    for v in initialRepresentatives:
                        updateGroupStats(v, visibleG, G)
                    updateInformationList(
                        initialRepresentatives, visibleG, groupsFound)
                    instancesSet = {}
                    graphSet = {}
                    if isLogEnabled():
                        if logInstancesEnabled & (reqestCount % loggingDeltaTime == 0):
                            instancesSet[reqestCount] = createPotentialInstances(
                                OPEN._open, visibleG, reqestCount)  # log before running
                        if logGraphEnabled & (reqestCount % loggingDeltaTime == 0):
                            graphSet[reqestCount] = createGraphInstance(
                                visibleG, reqestCount, targetID)
                    for v, k in OPEN:
                        CLOSED.add(v)
                        updateInformation(v, visibleG, groupsFound)
                        reqestCount = reqestCount+1
                        stats["profileID"] = v.strip()
                        stats["heuristic"] = k
                        stats["IsLeadRequestsNum"] = reqestCount
                        stats["notleadsFound"] = len(notLeads)
                        stats["closed"] = len(CLOSED)
                        stats["open"] = len(OPEN)
                        stats["groupsFound"] = len(groupsFound)
                        stats["time"] = time.time() - T
                        resultf.writerow(stats)
                    print OPEN.__class__.__name__+str(execution),  time.time() - T
                    if isLogEnabled():
                        if logInstancesEnabled & (reqestCount % loggingDeltaTime == 0):
                            # log after network updates
                            instancesSet[reqestCount] = createPotentialInstances(
                                OPEN._open, visibleG, reqestCount)
                        if logGraphEnabled & (reqestCount % loggingDeltaTime == 0):
                            graphSet[reqestCount] = createGraphInstance(
                                visibleG, reqestCount, targetID)
                        if logInstancesEnabled:
                            logInstances(targetID, OPEN,
                                         visibleG, instancesSet)
                        if logGraphEnabled:
                            logGraph(targetID, OPEN, graphSet)
                    print "Done."
                    pass
