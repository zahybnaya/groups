library('ggplot2')
d=read.csv('data.csv')
maxGroups = max(d$groupsFound)
maxRequestPerAlg=aggregate(d$IsLeadRequestsNum,list(d$alg),max)#Not working now... TODO
d$recall=d$groupsFound/maxGroups
d$resource=d$IsLeadRequestsNum/max(d$IsLeadRequestsNum) #TODO use per algorithm 
pdf('entire.pdf')
print(ggplot(data=d, aes(x=d$resource, y=d$recall, group=factor(d$alg))) +  geom_line(aes(linetype= factor(d$alg))) + geom_point(size=0) + ggtitle("Finding Entire Bipartite graph") + ylab(" % groups") + xlab("#flips") + scale_linetype_discrete(name="Method:") + theme_bw()+ theme(legend.position="top") + theme(legend.position="top"))
dev.off()
