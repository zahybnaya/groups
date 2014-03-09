
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

# Convert 2-mode network to 1-mode members network
# Example: http://www.stanford.edu/~messing/Affiliation%20Data.html
df <- data.frame( group = c('a','b','c','a','b','c','d','b','d'),
                  person = c('Sam','Sam','Sam','Greg','Tom','Tom','Tom','Mary','Mary'), 
                  stringsAsFactors = F)
plot(graph.data.frame(df, directed=F))
M = as.matrix( table(df) )
Mgrp = M %*% t(M)
g1m_grp <- graph.adjacency(Mgrp, weighted=T)
Mprsn <- t(M) %*% M
g1m_prsn <- graph.adjacency(Mprsn, weighted=T)
