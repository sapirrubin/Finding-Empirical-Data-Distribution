import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from numpy.random import randint
from numpy.random import rand
from scipy import stats

#Functions

#Finding power law alpha parameter using MLE
def estimate_alpha(x,x_min):
    n=len(x)
    sigma = 0
    for i in range(n):
        if x[i] > 0:
            sigma = sigma + math.log(float(x[i]/np.abs(x_min-0.5)))
    return (1+(n/sigma))

#KS implementaion for searching subsets    
def KS_test(x_sampels_for_ga,dis_to_fit):
    xmins = list(x_sampels_for_ga)
    xmins.sort()
    KS = []

    if dis_to_fit == 'normal':
        # estimate mu and sigma using direct MLE
        mu,sigma = stats._continuous_distns.norm.fit(xmins)           
        x_min = xmins[0]
        n = len(xmins)
        # construct the empirical CDF - steps function 1/n
        emp_cdf = list(np.arange(1, n+1) / n)
#         emp_cdf.sort(reverse= True)
        # construct the fitted theoretical CDF
        theo_cdf = stats._continuous_distns.norm.cdf(x=xmins,loc=mu,scale=sigma)
        KS.append(stats.ks_2samp(emp_cdf,theo_cdf)[0])
    else:
        for i,x_min in enumerate(xmins):
            # choose next xmin candidate
            # truncate data below this xmin value
            Z1 = [z for z in xmins if z>=x_min]
            n = len(Z1)
            # construct the empirical CDF - steps function 1/n
            emp_cdf = list(np.arange(1, n+1) / n)
            emp_cdf.sort(reverse= True)
            # construct the fitted theoretical CDF
            theo_cdf = []
            if dis_to_fit == 'pl':
            # estimate alpha using direct MLE
                a = estimate_alpha(x=Z1,x_min=x_min) 
                for z1 in Z1:
                    theo_cdf.append((z1/x_min)**(1-a))
            elif dis_to_fit == 'exp':
                # estimate lambda using direct MLE
                lamb = 1/((sum(Z1)/n)-x_min) 
                for z1 in Z1:
                    theo_cdf.append(1 - math.exp(lamb*(x_min-z1)))
            # compute the KS statistic
            abs_diff = []
            for k in range(n):
                abs_diff.append(abs(theo_cdf[k] - emp_cdf[k]))
            KS.append(max(abs_diff))

    # find minimum of KS
    #removing 0 for avoiding empty group
    KS_no_zero = [ks for ks in KS if ks>0]
    min_ks = min(KS_no_zero)
    xmin_index = KS.index(min_ks)
    
    return (min_ks, xmin_index)

#Gentic algorithem functions    
#decode bitstring to numbers
def decode(x_sampels, bitstring):
    decoded = []
    for index,bit in enumerate(bitstring):
        if bit:
            decoded.append(x_sampels[index])
    return decoded
 
# tournament selection
def selection(pop, scores, k=5):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
# mutation is removing one sample
def mutation(children, r_mut):
    for i,chromosone in enumerate(children):
        # check for a mutation
        if rand() < r_mut:
            # removing the point
            children.remove(chromosone)

#GA

def genetic_algorithm(objective,samples, dis_to_fit, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, len(samples)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best = pop[0]
    best_eval,best_eval_index = objective(decode(samples, pop[0]),dis_to_fit)
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(samples, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d,dis_to_fit)[0] for d in decoded]
        scores_index = [objective(d,dis_to_fit)[1] for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval, best_eval_index = pop[i], scores[i], scores_index[i]
                print(">%d, new best f() = %f" % (gen, scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [decode(samples, best), best_eval,best_eval_index]
    
def p_pl(x_i,x_min,alpha):
    if x_i >= x_min:
        return ((alpha -1)/x_min)*((x_i/(x_min))**(-alpha))
    else:
        return 0

#main function
#receving:
#1) the desire distribution to find
#2) the empirical data
#1) the empirical data distribtuion if exist, usually for searching power law distribtuion for degree disribution in a graph
def find_empirical_dist(disribution, samples, emprical_dist=[]):
    # define the total iterations
    n_iter = 100
    # define the population size
    n_pop = 10
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(len(samples))

    # perform the genetic algorithm search
    best, score, x_min_index = genetic_algorithm(objective=KS_test, samples=samples, dis_to_fit=disribution ,
                                                 n_iter=n_iter, n_pop=n_pop, r_cross=r_cross, r_mut=r_mut)
    print('Done!')
    print('f() = %f' % (score))
    result_x_min = best[0]
#     print(f'estimate is {result_x_min}')
    if disribution == 'pl':
        result_alpha = estimate_alpha(x=best,x_min=result_x_min)
        print(f'alpha estimate is {result_alpha}')
        print(f'length of p_pl found is {len(best)}')
        print(f'length original found is {len(samples)}')
    elif disribution == 'exp':
        result_lambda = 1/((sum(best)/len(best))-result_x_min)
        print(f'lamnda estimate is {result_lambda}')
        print(f'length of p_exp found is {len(best)}')
        print(f'length original found is {len(samples)}')
    elif disribution == 'normal':
        result_mu, result_sigma = stats._continuous_distns.norm.fit(best) 
        print(f'mu estimate is {result_mu} and sigma estimate is {result_sigma}')
        print(f'length of p_normal found is {len(best)}')
        print(f'length original found is {len(samples)}')

    if disribution == 'pl':
        p_x=[]
        for x_i in best:
            p_x.append(p_pl(x_i=x_i, x_min=result_x_min, alpha=result_alpha))
        p_x.sort(reverse=True)

        fig=plt.figure()

        ax=fig.add_subplot(111, label="1")
        ax2=fig.add_subplot(111, label="2", frame_on=False)
        
        ax.plot(best, p_x, color = 'orange', label = 'fitted data')
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.set_xlabel('', color = 'orange')  
        ax.set_ylabel('', color = 'orange')     
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        
        if not emprical_dist.empty:
            ax2 = sns.distplot(emprical_dist,kde=True,label='observed data')
        else:
            ax2 = sns.distplot(samples,kde=True,label='observed data')
        ax2.tick_params(axis='x')
        ax2.tick_params(axis='y', colors="b")
        ax2.xaxis.set_label_position('top') 
        ax2.yaxis.set_label_position('right')
        ax2.set_xlim(0,)
        ax.legend(loc=1)
        ax2.legend(loc=3)
        plt.show()
        
    elif disribution == 'exp':
        sns.distplot(samples,kde=True,label='observed data')
        plt.plot(best, stats._continuous_distns.expon.pdf(best, loc=0, scale=(1/result_lambda)) ,label = 'fitted data')
        plt.legend()
        plt.xlim(0,)
        plt.show()
    elif disribution == 'normal':
        sns.distplot(samples,kde=True,label='observed data')
        plt.plot(best, stats._continuous_distns.norm.pdf(best, loc=result_mu, scale=result_sigma) ,label = 'fitted data')
        plt.legend()
        plt.xlim(0,)
        plt.show()

    if disribution == 'pl':
        n_cdf = len(best)
        y_cdf = list(np.arange(1, n_cdf+1) / n_cdf)
        y_cdf.sort(reverse= True)
        scipy_ks_test, scipy_p_value = stats.ks_1samp(y_cdf,stats._continuous_distns.powerlaw.cdf,args=(result_alpha,0))
    elif disribution == 'exp':
        scipy_ks_test, scipy_p_value = stats.ks_1samp(best,stats._continuous_distns.expon.cdf,args=(0,1/result_lambda))
    elif disribution == 'normal':
        scipy_ks_test, scipy_p_value = stats.ks_1samp(best,stats._continuous_distns.norm.cdf,args=(result_mu,result_sigma))
    print(f'KS results are ks_test : {scipy_ks_test} and p_value : {scipy_p_value}')
    return (best)
