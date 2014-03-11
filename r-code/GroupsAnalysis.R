
# Load igraph
library(igraph)

wdstr <- "C:\\Users\\Ofrit\\Dropbox\\groups\\code"
setwd(wdstr)

## Read the group b-partite comma-separated value file
vrtx <- read.csv(paste(wdstr, "\\ning_data\\NingGroups.csv",sep=""))
g <- graph.data.frame(vrtx, directed=F)
g_base <- graph.data.frame(vrtx, directed=F)

# make it a bipartite graph
# See http://stackoverflow.com/questions/15367779/how-to-create-a-bipartite-network-in-r-with-igraph-or-tnet 
V(g)$type <- V(g)$name%in% vrtx[,1]
is.bipartite(g)

# There ar 2 clusters in Ning
no.clusters(g)
gc <- clusters(g)

# Names of group & members not in giant component
V(g)[which(gc$membership == 2)]
# Remove nodes that do not belong to giant component
g <- delete.vertices(g, which(gc$membership == 2))
degree(g)

# Groups per profile staristics
summary(degree(g)[which(V(g)$type == FALSE)])
# Members per group statistics
summary(degree(g)[which(V(g)$type == TRUE)])

# Function - create m-graph
# m-graph is a 1-mode groups graph created from the group-user edgelist


# Sandbox - should be redone for all datasets

# Convert 2-mode network to 1-mode members network
# Example: http://www.stanford.edu/~messing/Affiliation%20Data.html
# df <- data.frame( group = c('a','b','c','a','b','c','d','b','d'),
#                   person = c('Sam','Sam','Sam','Greg','Tom','Tom','Tom','Mary','Mary'), 
#                   stringsAsFactors = F)
df <- data.frame( group = c('a','b','c','a','b','c','b','c','d','b','d'),
                  person = c('Sam','Sam','Sam','Greg', 'Greg','Greg','Tom','Tom','Tom','Mary','Mary'), 
                  stringsAsFactors = F)
plot(graph.data.frame(df, directed=F))
M = as.matrix( table(df) )
Mgrp = M %*% t(M)
g1m_grp <- graph.adjacency(Mgrp, weighted=T, mode="lower")
E(g1m_grp)$weight
g1m_grp <- simplify(g1m_grp)
plot(g1m_grp)

# Mprsn <- t(M) %*% M
# g1m_prsn <- graph.adjacency(Mprsn, weighted=T)
# g1m_prsn <- graph.adjacency(Mprsn, weighted=T, mode="lower")
# plot(g1m_prsn)
# plot(simplify(g1m_prsn))

# List of all cliques size 3 or larger
clq <- cliques(g1m_grp, min=3)
length(clq)

# For each clique - find (1) clique size (2) number of profiles in it
grpClique <- data.frame(matrix(nrow=0, ncol=2))


for (i in 1:length(clq) )
{
  nodes <- clq[[i]]
  clqSubGraph <- subgraph.edges(g1m_grp, nodes)
  
  # get.edge.ids - may be able tyo use this, but need a paiwise vids
  
  # since this is a 1-mode graph with weigthts, we can find the number of common members in all clique groups
  # Note: the weights indeiate the number of common members between the groups (vertecies attached to the edge)
  # We take the minimun among all edges in the clique to identify the number of common members
  num_of_members_in_groups <- min(E(clqSubGraph)$weight)
  clqLine <- c(length(nodes),num_of_members_in_groups)
  grpClique <- rbind(grpClique, clqLine)
}







# Add the column labels
colnames(grpClique) <- c("clqSize", "profilesNum")

# clique.number(g1m_grp)

# largest.cliques(g1m_grp)
# maximal.cliques(g1m_grp)

# Find cliques

g <- erdos.renyi.game(100, 0.3)
clique.number(g)
cliques(g, min=6)
largest.cliques(g)
maximal.cliques(g)
