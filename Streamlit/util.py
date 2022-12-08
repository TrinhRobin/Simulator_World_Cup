
import requests
from io import StringIO
import csv
import pandas as pd
from itertools import product
import seaborn as sns
import numpy as np
import time
import matplotlib.pyplot as plt

def feature_engineering(dataframe_elo_score,dataframe_game_results,date_bounds=['2018',"2022-11-20"],
                        tournaments_list=['Friendly', 'FIFA World Cup qualification', 'UEFA Nations League',
       'UEFA Euro qualification', 'African Cup of Nations qualification',
       'CONCACAF Nations League', 'African Cup of Nations',
       'African Nations Championship', 'COSAFA Cup',
       'AFC Asian Cup qualification', 'Gold Cup', 'FIFA World Cup',
       'Copa América', 'UEFA Euro', 'AFC Asian Cup',
       'CONIFA World Football Cup', 'Island Games',
       'CONCACAF Nations League qualification',
       'African Nations Championship qualification', 'AFF Championship',
       'CONIFA European Football Cup', 'CECAFA Cup', 'EAFF Championship',
       'CFU Caribbean Cup qualification', 'SAFF Cup', 'Arab Cup', 'Gulf Cup',
       'Pacific Games', 'Kirin Challenge Cup',
       'Inter Games Football Tournament', 'Confederations Cup',
       'Oceania Nations Cup', 'Pacific Mini Games', 'Intercontinental Cup',
       'Baltic Cup', 'AFC Challenge Cup', "King's Cup",
       'Gold Cup qualification', 'Windward Islands Tournament', 'UNCAF Cup']):
    """
    Inputs : 
        dataframe_elo_score = pandas Dataframe with list of teams, their elo_score rating), rank, and various informations
        dataframe_game_results = pandas Dataframe with results of games: Home Team, Away Team, Goal Home, Goal Away .... (from kaggle)
        tournment_list = List of the tournaments that will remain in the final dataframe, the other ones (mostly minor competitions will not be considered)
    Output : we filter, clean and compute additional features,mainly the XG of each team, and add these additional features by merging the two dataframes
    """
    
    #filter by date
    dataframe_game_results2 =dataframe_game_results[(dataframe_game_results['date']>=date_bounds[0])
                                                    &(dataframe_game_results['date']<date_bounds[1])]
    #filter tournaments
    dataframe_game_results2  = dataframe_game_results2[dataframe_game_results2['tournament'].isin(tournaments_list)]
    
    #We create all the couples Team A vs Team B for Home Game:
    df1 = dataframe_game_results2[['home_team','home_score']]
    df1 = df1.rename(columns={'home_team':'team','home_score':'score'})
     #We create all the couples Team A vs Team B for Away Game:
    df2 = dataframe_game_results2[['away_team','away_score']]
    df2= df2.rename(columns={'away_team':'team','away_score':'score'})
    
    #Concate
    df_xg1 = pd.concat([df1,df2],axis=0)
    df_xg1 =df_xg1.reset_index(drop=True)
    #Are there any NaN values?
    df_xg1.isna().sum()
    #dropping NaN values #or replacing with fillna(0)?
    df_xg1 = df_xg1.dropna()
    #Checking that there are not nan Values remaining
    df_xg1.isna().sum()
    #Computing the XG = mean goals over the last games 
    df_extract = df_xg1.groupby('team').agg(score = ('score','mean'),
                                             nb_match= ('score','count'))
    #Merging The two dataframes
    df_features =dataframe_elo_score.merge(df_extract ,how='inner',left_on='Team_Name',right_on='team')

    df_couples = pd.DataFrame(list(product(dataframe_elo_score['Team_Name'], dataframe_elo_score['Team_Name'])),columns=['Team_A','Team_B'])
    #Removing the couples Team A vs Team B where they are the same
    df_couples=df_couples[df_couples['Team_A']!= df_couples['Team_B']]
    df_couples = df_couples.merge(df_features[['Rank_Team','Rating','Team_Name','score']],left_on='Team_A',right_on='Team_Name')
    df_couples = df_couples.merge(df_features[['Rank_Team','Rating','Team_Name','score']],left_on='Team_B',right_on='Team_Name',suffixes=('_A', '_B'))
    #suffixes=('_x', '_y')
    df_couples=df_couples.drop( columns=['Team_Name_A','Team_Name_B'])
    #Computing the TWO MAIN FEATURES (which we will used in our model) : XG RATE difference (between team A and team B)
    #And ELO Rating difference = Difference of Rank/Power between Team A and Team B
    df_couples['XG_difference'] = df_couples['score_A'] - df_couples['score_B']
    df_couples['Rating_difference']=df_couples['Rating_A'] - df_couples['Rating_B']
    return(df_couples)


def compute_XG_diff_mean_std(dataframe,equipe1,equipe2,bound=0.25):
    """
    Input : dataframe : pandas DataFrame, the result from the feature_engineering function
    equipe1, equipe 2 = str , names of Team A and Team B
    bound = float between 0 and 1, Determines the range of the sample (from which we extract the mean and the standard deviation)
    Output : Computing the XG difference between equipe1 and equipe2
    By sampling mean and standard deviation from a subset of dataframe,( = the games between teams about the same ELO ratings difference as equipe1 and equipe1)
    (Hypothesis : The XG_differences between teams with about the same elo ratings follow a normal distribution)
    """
    #A Team cannot play against itself 
    if equipe1 == equipe2:
        return("Error : same team")
    elo_diff = dataframe.loc[(dataframe['Team_A']==equipe1)
                             &(dataframe['Team_B']==equipe2),'Rating_difference'].values
    #Predict XG_difference :
    #mean = model.predict(elo_diff.reshape(-1, 1) 
    #We select a subset of the historical games based on the parameter bound (those between Teams whose Elo Ratings Difference is about the same as the two teams in inputs) from which we sample the mean and the standard
    #print(elo_diff)
    sample = dataframe.loc[(dataframe['Rating_difference']<=max(elo_diff[0]*(1+bound),elo_diff[0]*(1-bound)))&
                      ( dataframe['Rating_difference']>=min(elo_diff[0]*(1+bound),elo_diff[0]*(1-bound))),'XG_difference']
    #Computing the mean and the standard deviation from the dataframe's subset
    sample_mean = sample.mean()
    sample_std  =sample.std()
    return(sample_mean,sample_std)

#TO DO
#Add best player field, XG_best_player and injury 
#At each game we draw a random die to determine an injury event 
#if yes we remove the XG player to the XG_team
class Team:
    """
    A football Team
    Attributes :
        name : str
        xg : float Expected Goals (how many goals this team is expected to score)
        elo_ranking : float ELO represents the strength/reputation of the Team
        results : dictionnary which stores this teams results
    Methods :
        init
        print
    """
    def __init__(self, team_name,XG ,elo_ranking):
            self.name = team_name
            self.xg = XG
            self.elo_ranking = elo_ranking
            self.results = {"Win":0,"Draw":0,"Loss":0,"Goals":0,"Goals_Against":0,"Points":0}
    def __str__(self):
        return(f"Team {self.name}: \n XG : {self.xg} \n ELO Ranking : {self.elo_ranking} \n Current Results : {self.results}")


class Match:
    """
    Simulate a Football Game between two Teams
    Attributes :
        team A: Team The first Team
        team B : Team The opponent Team
        diff_xg_mean : Float difference of XG rates of the two Teams
        is_round_robin : Boolean whether or not This match is from the group stage (True) or from the Final Stage (False)
        winner = Team  The winning Team
    Methods :
        init
        print
        simulate_match : Sample N_samples *2 Poisson Laws and average the result for computing probabilities of Win/Draw/Loss
    """
    def __init__(self, team_A, team_B,dataframe, is_round_robin,N_samples=1000):
        self.team_A = team_A
        self.team_B = team_B
        res = compute_XG_diff_mean_std(dataframe, team_A.name, team_B.name,bound=0.25)
       # print(res)
        self.diff_xg_mean , self.diff_xg_std = res[0],res[1]
        self.is_round_robin = is_round_robin
        self.winner = None
        #probabilities of W/D/L
        self.results = [0,0,0]
        self.simulate_match(N_samples)

    def simulate_match(self,N_samples):
        #Computing Elo Ranking difference
        #Predict XG_difference :
        #do multiple sampling?
        XG_difference = np.random.normal(loc=self.diff_xg_mean, scale=self.diff_xg_std)
        #print(XG_difference)
        #changing Xg_difference by  weighting more recent matchs and select only ranking teams accordingly_adversaries
        new_XG_team_A, new_XG_team_B =max(0,self.team_A.xg+XG_difference/2),max(0,self.team_B.xg-XG_difference/2)
        
        #Simulate 100000 matchs :
        simul_results =[[np.random.poisson(new_XG_team_A), np.random.poisson(new_XG_team_B)]for i in range(N_samples)]
    
        self.results = [np.mean([ score[0]>score[1] for score in simul_results]), 
                        np.mean([ score[0]==score[1] for score in simul_results]),
                       np.mean([ score[0]<score[1] for score in simul_results]) ]
       # print(   self.results)
        
        if np.argmax(self.results)== 0:
            if self.is_round_robin :
                self.team_A.results["Points"]+=3
                self.team_B.results["Points"]+=0
            self.team_A.results["Win"]+=1
            self.team_B.results['Loss']+=1
            self.winner = self.team_A
            
        elif np.argmax(self.results)== 2:
            if self.is_round_robin :
                self.team_A.results["Points"]+=0
                self.team_B.results["Points"]+=3
            self.team_A.results["Loss"]+=1
            self.team_B.results['Win']+=1
            self.winner = self.team_B
        else :
            if self.is_round_robin :
                self.team_A.results["Points"]+=1
                self.team_B.results["Points"]+=1
            elif not self.is_round_robin :
                #During the Final Stage ->Penalty shoothout = Head or Tail 
                self.winner = np.random.choice([self.team_A,self.team_B])
            self.team_A.results["Draw"]+=1
            self.team_B.results['Draw']+=1
            
        #To Do : Adding int() to round these quantitie and have integer values for number of goals
        self.team_A.results['Goals']=  np.mean([ score[0] for score in simul_results])
        self.team_A.results['Goals_Against']= np.mean([ score[1] for score in simul_results])
        self.team_B.results['Goals']= np.mean([ score[1] for score in simul_results])
        self.team_B.results['Goals_Against']= np.mean([ score[0] for score in simul_results])
  
    def __str__(self):
        return ("\n"+self.team_A.name + " Probability of winning is " +str(100*self.results[0])+" % \n " +
        "Probability of Draw is "+str(100*self.results[1] )  +"%\n"+
        self.team_B.name + " Probabilty of winning is " +str(100*self.results[2])+" %  ")

# The winner of the group stage is obtained from
# 1 - points
# 2 - goal difference
# 3 - goal scored
# 4 - Random Sample      

class Group_Stage:
    """
    A Group Stage between 4 Teams = 2*3 Match
    Attributes :
        first_qualified : Team Winner of the Group Stage
        second_qualified : Team The Team whose final rank is 2
        teams = list the liste of the 4 Teams of the Group
    Methods:
        reset : initialize the games results (= beginning of the tournament)
        play_group_stage : play the 3 games for each Teams and rank them to see which Teams has qualified for the next round
    """
    def __init__(self, teams,dataframe):
        self.first_qualified = None
        self.second_qualified = None
        self.teams = teams
        self.reset()
        self.play_group_stage(dataframe)
    def reset(self):
        for team in self.teams:
            self.results = {"Win":0,"Draw":0,"Loss":0,"Goals":0,"Goals_Against":0,"Points":0}

    def play_group_stage(self,dataframe):
        [Match(self.teams[i], self.teams[j],dataframe, True,1000) for i in range(0, len(self.teams))  for j in range(i + 1, len(self.teams))]
        #Sorting the teams           
        self.teams.sort(key= lambda elem : (elem.results['Points'],elem.results['Goals']-elem.results['Goals_Against'],elem.results['Goals'],
                                            np.random.rand()) ,reverse=True)
         #Store the two qualified Teams
        self.first_qualified = self.teams[0]
        self.second_qualified = self.teams[1]

class Tournament :
    """
    Simulate the whole Tournament !
    Attributes :
        all_teams = list of 32 Teams = participants to the Tournmanent
        groups = list of 8 group stages
        data = pandas Dataframe we store the data necessary to computing the XG difference betwee each teams
    method :
        init
        print
        simul_one_tournament : simulate one tournament: Group Stage and Final Stage
        main : Iterates simul_one_tournament n times 
    """
    def __init__(self,dataframe,groups_list):
        #dictionary {team_name str : team Team Object}
        self.all_teams = { team : Team(team, dataframe.loc[dataframe['Team_A']==team,'score_A'].mode()[0],
         dataframe.loc[dataframe['Team_A']==team,'Rating_A'].mode()[0] ) for team in dataframe['Team_A'].unique()}
        self.groups = groups_list
        self.data = dataframe

    def simul_one_tournament(self,winners):
        # Play round robin
        #Simulate all Group stages
        Groups =[Group_Stage([ self.all_teams[teams_group] for teams_group in group],self.data) for group in self.groups]

        # Play final elimination  phase 
           
        huitiemes = [(Match(Groups[2*i].first_qualified,Groups[2*i+1].second_qualified,
                                self.data,
                                False ,1000),
                          Match(Groups[2*i].second_qualified,Groups[2*i+1].first_qualified,self.data ,False,1000))
                         for i in range(int(len(self.groups)/2))]
             # Quarters

        quarters = [Match(quart_finalist[0].winner,quart_finalist[1].winner,self.data,False ) for quart_finalist in huitiemes ]
           
            # Semifinals
        semis = [Match(quarters[2*i].winner,quarters[2*i+1].winner,self.data,False,1000 ) for i in range(2)]
            # Final
        winner_final = Match(semis[0].winner ,semis[1].winner,self.data, False,1000)
        winner =winner_final.winner

        if winner.name in winners.keys():
            winners[winner.name] += 1
        else:
            winners[winner.name] = 1
            
    def main(self,n=10):
        winners ={}
        [self.simul_one_tournament(winners) for i in range(n)]
        for key in sorted(winners, key=winners.get, reverse=True):
            print (key + ": " + str(winners[key]))
        return(winners)