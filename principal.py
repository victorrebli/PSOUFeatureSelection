from importlib import reload
from decimal import Decimal
from random import sample, uniform, randint
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import problem
reload(problem)
import evaluate
reload(evaluate)
from problem import Problem
from evaluate import SolutionEvaluator


def log():
    logFormatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    
    if not rootLogger.handlers:
        fileHandler = logging.FileHandler(datetime.now().strftime('PSO_%d-%m_%H:%M.log'))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    return rootLogger

rootLogger  = log()  

def minimize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution < global_solution

def maximize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution > global_solution

def create_population_uniform_strategy(X, num_particles):
    
    _, n_cols = X.shape
    lower = int((num_particles / 3) * 2)
    
    particles = np.zeros(shape=(num_particles, n_cols + 1))
    for i in range(lower):
        features = sample(range(n_cols), int(round(n_cols*0.2)))
        
        for j in features:
            particles[i,j] = round(Decimal(uniform(0.61,1.0)), 4)
            
    for i in range(lower, num_particles):
        qtd_high = sample(range(n_cols), randint(round(n_cols/2 + 1), n_cols))
        
        for j in qtd_high:
            particles[i,j] = round(Decimal(uniform(0.61, 1.0)),4)
            
    return particles


def create_population_20_50_strategy(X, num_particles):
    
    _, n_cols = X.shape
    particles = np.zeros(shape=(num_particles, n_cols + 1))    
    lower_group = int((num_particles / 3 ) * 2)
    
    for i in range(lower_group):
        features = sample(range(n_cols), int(round(n_cols * 0.2)))
    for j in features:
            particles[i,j] = round(Decimal(uniform(0.61,1.0)), 4)
    
    for i in range(lower_group, num_particles):
        features = sample(range(n_cols), randint(round(n_cols / 2 + 1), n_cols))
        for j in features:
             particles[i,j] = round(Decimal(uniform(0.61, 1.0)), 4)
                                                 
    return particles                                             
                                                 
class PSOException(Exception):
                                                 
     def __init__(self,message):
         super(PSOException, self).__init__()
         self.message = message
                                                 
     def __str__(self):
         
         return repr(self.message)                                        

class PSOSelector(object):
      
       def __init__(self, estimator, w=0.7298, c1=1.49618, c2=1.49618,
                    num_particles=30, max_iter=100, max_local_improvement=50,
                    maximize_objective=True, initialization='uniform',
                    fitness_method='type_2', cv = 3, verbose=True):
                                                 
           self.w = w
           self.c1 = c1
           self.c2 = c2
           self.num_particles = num_particles
           self.max_iter = max_iter
           self.cv = cv
           self.evaluator_ = None
           self.estimator = estimator
           self.velocity_ = None
           self.best_individual_ = None
           self.best_global_ = None
           self.best_global_fm = 0
           self.best_global_cp = 0
           self.best_cp_ = 0
           self.best_fm_ = 0
           self.solution_ = None
           self.initialize = 0
           self.initialize_1 = 0
           self.verbose = verbose
           self.N = None
           self.max_local_improvement = max_local_improvement
           self.local_improvement = 0
           self.particles = None
           self.count = []
           self.N_ = 0
           self.iteration_ = 0
           self.pop_ = {}
           self.count_global = 0
           self._setup_initialization(initialization, fitness_method)
           self._setup_solution_comparator(maximize_objective)
           self.selected_features_ = None
           
       def _setup_initialization(self, initialization, type_search):
           
           init_method = {
                   'uniform': create_population_uniform_strategy,
                   '20_50': create_population_20_50_strategy
                         }
           
           init_search = {
                    'type_1': self.search_type_1,
                    'type_2': self.search_type_2
                          }     
                                                     
           self._initialization = initialization
           if initialization not in init_method:
               raise PSOException(f'Invalid method {initialization!r}')
           self.init_method_ = init_method[initialization]
           self.init_search_ = init_search[type_search]
           
       def _setup_solution_comparator(self, maximize_objective):
           
           self.maximize_objective = maximize_objective
           if self.maximize_objective:
               self.is_solution_better = maximize_comparator
           else:
               self.is_solution_better = minimize_comparator
               
               
               
       def fit(self, X, unused_y = None, **kargs):
           
           if not isinstance(X, pd.DataFrame):
               raise PSOException('The "X" parameter must be a data frame')
        
           colunas_full  = X.columns
            
           self._initialize(X)
           

           if unused_y.all() != None:
               self.pop_['cp'] = np.zeros(shape=(1, self.num_particles))[0]
               self.pop_['fm'] = np.zeros(shape=(1, self.num_particles))[0]

           prob = Problem(X, unused_y, self.estimator,
                          self.cv, **kargs)
           
           self.N_ = prob.n_cols
           self.evaluator_ = SolutionEvaluator(prob, self.num_particles)
           
           self.velocity_ = np.zeros(shape=(self.num_particles, self.N_))
           self.best_individual_, self.best_individual_[:] = \
             np.zeros(shape=(self.num_particles, self.N_ + 1)), 'nan'
        
           self.best_global_, self.best_global_[:] = \
             np.zeros(shape=(1, self.N_ + 1)), 'nan'
           self.solution_ = np.zeros(shape=(self.max_iter + 1, self.N_ + 1))
           
           #Parameters to store the purity and fmeasure metric
           self.best_fm_ = np.zeros(shape=(1, self.num_particles))[0]
           self.best_cp_ = np.zeros(shape=(1, self.num_particles))[0]

           while not self._is_stop_criteria_accepted():
               self.init_search_()
               count_sel_feat = self.count_features(self.best_global_[0])
               
               best_glob = self.best_global_[0]
               self.selected_features_  = np.ma.masked_where(best_glob[:-1]>0.6, best_glob[:-1])
               self.selected_features_, = np.where(self.selected_features_.mask == True)
               colunas = colunas_full[self.selected_features_]

               if self.verbose:
                   #breakpoint()
                   interm_var = f'Iteration: {self.iteration_}/{self.max_iter} \n ,'
                   interm_var = interm_var + f'Best global metric - CP X PF: {self.best_global_[:, -1]} \n , '
                   if unused_y.all() != None:

                       interm_var = interm_var + f'Best global metric purity: {self.best_global_cp} \n ,'
                       interm_var = interm_var + f'Best global metric f-measure: {self.best_global_fm} \n ,'     

                   interm_var = interm_var + f'Index features_selected: {self.selected_features_} \n , '
                   interm_var = interm_var + f'Number of selected features: {count_sel_feat} \n , ' 
                   interm_var = interm_var + f'Columns selected: {colunas}'       
                   rootLogger.info(interm_var)
              
               
               for i in range(0, self.num_particles):
                   self.count.append(self.count_features(
                           self.best_individual_[i, :]))
                       
               best_glob = self.best_global_[0]
               self.selected_features_ = np.ma.masked_where(best_glob[:-1]>0.6, best_glob[:-1])
               self.selected_features_, = np.where(self.selected_features_.mask == True)
               
       def _initialize(self, X):
           
          self.iteration_ = 0
          self.pop_['pop'] = self.init_method_(X, self.num_particles)
            
       def _is_stop_criteria_accepted(self):
            
          no_global_improv = self.local_improvement >= self.max_local_improvement
          max_iter_reached = self.iteration_ >= self.max_iter
          return max_iter_reached or no_global_improv
      
       def search_type_1(self):
          
          self.pop_ = self.evaluator_.evaluate(self.pop_)
          self.calculate_best_individual_pso_1_1(self.pop_)
          self.calculate_best_global_pso_1_1()
          self.solution_[self.iteration_, :] = self.best_global_
          self.update_velocity()
          self.iteration_ += 1
          
          
       def search_type_2(self):
         
          self.pop_ = self.evaluator_.evaluate(self.pop_)
          self.calculate_best_individual(self.pop_)
          self.calculate_best_global()
          self.solution_[self.iteration_, :] = self.best_global_
          self.update_velocity()
          self.iteration_ += 1
         
       def update_velocity(self):
         
         w = self.w
         c1, c2 = self.c1, self.c2
         
         for i in range(0, len(self.pop_['pop']) - 1):
             for j in range(0, self.N_):
                 r1= round(uniform(0,1), 2)
                 r2 = round(uniform(0, 1), 2)
                 pop = self.pop_['pop'][i, j]
                 inertia = w * self.velocity_[i,j]
                 cognitive = c1 * r1 * (self.best_individual_[i,j] - pop)
                 social = c2 * r2 * (self.best_global_[0, j] - pop)
                 velocity = inertia + cognitive + social
                 self.velocity_[i,j] = velocity
                 self.pop_['pop'][i,j] += velocity
                 
       def calculate_best_individual(self, pop):

         if self.initialize == 0:
             for i in range(0, len(pop['pop'])):
                 for j in range(0, self.N_ + 1):
                     self.best_individual_[i,j] = pop['pop'][i, j]
                 if 'fm' in pop.keys():
                     self.best_fm_[i] = pop['fm'][i]
                 if 'cp' in pop.keys():
                     self.best_cp_[i] = pop['cp'][i]
             self.initialize = 1
             return             

         for i in range(0, len(pop['pop'])):
             candidate_a = pop['pop'][i, self.N_]
             candidate_b = self.best_individual_[i, self.N_]
             if self.is_solution_better(candidate_a, candidate_b):
                 for j in range(0, self.N_ + 1):
                     self.best_individual_[i, j] = pop['pop'][i,j]
                 if 'fm' in pop.keys():
                     self.best_fm_[i] = pop['fm'][i]
                 if 'cp' in pop.keys():
                     self.best_cp_[i] = pop['cp'][i]
                 continue               
         
             
             particle_count = self.count_features(self.pop_['pop'][i, :])
             count_best_individual = self.count_features(
                     self.best_individual_[i, :])
             
             if particle_count > 0:
                 if (pop['pop'][i,self.N_] == self.best_individual_[i, self.N_]
                        and particle_count < count_best_individual):
                     
                     for j in range(0, self.N_ + 1):
                         self.best_individual_[i,j] = pop['pop'][i,j]
                 if 'fm' in pop.keys():
                     self.best_fm_[i] = pop['fm'][i]
                 if 'cp' in pop.keys():
                     self.best_cp_[i] = pop['cp'][i]

                         
                         
       def calculate_best_global(self):

         if self.initialize_1 == 0:
             
             for i in range(0, self.N_ + 1):
                 self.best_global_[0,i] = self.best_individual_[0, i]
             if 'fm' in self.pop_.keys():
                 self.best_global_fm = self.best_fm_[0]
             if 'cp' in self.pop_.keys():
                 self.best_global_cp = self.best_cp_[0]

             self.initialize_1 = 1
             
             self.count_global = self.count_features(self.best_global_[0, :])
                         
         for i in range(0, self.num_particles):   
             if self.is_solution_better(self.best_individual_[i, self.N_],
                                        self.best_global_[0, self.N_]):
                 
                 self.local_improvement = 1
                 self.count_global = 0
                 
                 for j in range(0, self.N_ + 1):
                     self.best_global_[0,j] = self.best_individual_[i,j]

                 if 'fm' in self.pop_.keys():
                     self.best_global_fm = self.best_fm_[i]
                 if 'cp' in self.pop_.keys():
                     self.best_global_cp = self.best_cp_[i]    
                 self.count_global = self.count_features(
                         self.best_global_[0, :])
                 
                 continue
             
             count_best_individual = self.count_features(self.best_individual_[i, :])
             
             best_global = self.best_global_[0, self.N_]
             best_ind = self.best_individual_[i, self.N_]
             
             if (best_global == best_ind
                       and count_best_individual < self.count_global):
                 
                 self.local_improvement = 1
                 self.count_global = 0
                 for j in range(0, self.N_ + 1):
                     self.best_global_[0, j] = self.best_individual_[i,j]

                 if 'fm' in self.pop_.keys():
                     self.best_globao_fm = self.best_fm_[i]
                 if 'cp' in self.pop_.keys():
                     self.best_global_cp = self.best_cp_[i]

                 self.count_global = self.count_features(
                         self.best_global_[0, :])
                 
         self.local_improvement += 1        
                 
                 
       def calculate_best_individual_pso_1_1(self,pop):

         if self.initialize == 0:
             for i in range(0, len(pop['pop'])):
                 for j in range(0, self.N_ + 1):
                     self.best_individual_[i,j] = pop['pop'][i,j]
                 if 'fm' in pop.keys():
                     self.best_fm_[i] = pop['fm'][i]
                 if 'cp' in pop.keys():
                     self.best_cp_[i] = pop['cp'][i]


             self.initialize = 1
             return
         
         for i in range(0, len(pop['pop'])):
             if self.is_solution_better(pop['pop'][i,self.N_],
                                        self.best_individual_[i, self.N_]):
                 
                 for j in range(0, self.N_ + 1):
                     self.best_individual_[i, j] = pop['pop'][i,j]
                 if 'fm' in pop.keys():
                     self.best_fm_[i] = pop['fm'][i]
                 if 'cp' in pop.keys():
                     self.best_cp_[i] = pop['cp'][i]        
                    
                 
       def calculate_best_global_pso_1_1(self):
          if self.initialize_1 == 0:
              for i in range(0, self.N_ + 1):
                  self.best_global[0,i] = self.best_individual_[0,i]
                    
              if 'fm' in self.pop_.keys():
                   self.best_global_fm = self.best_fm_[0]
              if 'cp' in self.pop_.keys():
                   self.best_global_cp = self.best_cp_[0]        
              self.initialize_1 = 1    

          for i in range(0, len(self.pop_['pop'])):
              if self.is_solution_better(self.best_individual_[i, self.N_],
                                         self.best_global_[0, self.N_]):
                  self.local_improvement = 1
              
              for j in range(0, self.N_ + 1):
                  self.best_global_[0,j] = self.best_individual_[i,j]

              if 'fm' in self.pop_.keys():
                  self.best_global_fm = self.best_fm_[i]
              if 'cp' in self.pop_.keys():
                  self.best_global_cp = self.best_cp[i]        
          self.local_improvement += 1
          
          
       def count_features(self, particle_proportions, threshold=0.6):
          
         count = 0
         for i in range(0, self.N_):
             if particle_proportions[i] > threshold:
                 count = count + 1
         return count        
         
          
                  
              
              
              

            
                 
                 
                 
                
                
                
                
                
                
                
                
                     


            
                 
                 
                 
                 
                 
                 
                 
                 
                 
         
         
         
         
          
          
          
          
          
          
          
          
          
          
          
          
          
          
        
          
          
          
          
          
           
           
               
               
               
               
               
               
               
               
               
               
           
           
           
           
           
           
           
           
           


           
           
           
           
           
                                                 
                    
                    
                    
                                                 
                                                 