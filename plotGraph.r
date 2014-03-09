
ggplot(data=d, aes(x=d$IsLeadRequestsNum, y=d$groupsFound, group=factor(d$alg))) +  geom_line(aes(linetype = factor(d$alg)))
