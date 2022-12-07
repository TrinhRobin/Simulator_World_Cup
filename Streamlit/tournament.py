import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util import feature_engineering,Team, Match, Group_Stage

#Load data 
df_results = pd.read_csv("../data/results.csv")
df_elo = pd.read_csv("../data/elo_ratings_team.csv", sep='\t',index_col=0)
df_elo =df_elo.replace({'United States\tUSA':'United States'})
df_test = feature_engineering(df_elo,df_results)

st.sidebar.title("Sidebar")
pages = ["Project Presentation","Data Viz","Modeling Match","Modeling Group Stage","Modeling the whole Tournament"]
page = st.sidebar.radio("Choose a page",options =pages)

if page == pages[0]:
    st.header("World Cup Simulator")
    st.title("For the Demo Streamlit Masterclass")
    st.subheader('By the Datascientest team')
    st.image('Fifa_WC_2022.png')
    st.markdown("From the original project from [A Mathematician Guide from the World Cup](https://youtu.be/KjISuZ5o06Q)")
    st.markdown("You may find more details about the implementation on this [Github Account](https://github.com/TrinhRobin/Simulator_World_Cup)")
    st.video("https://youtu.be/KjISuZ5o06Q")
    st.write("Some insights about the dataframe used in this project:")
    st.dataframe(df_test.head())
if page == pages[1]:
    st.header("Hypothesis 1 : The XG difference depends on ratings difference")
    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        ax = sns.scatterplot(data=df_test,x='Rating_difference',y='XG_difference')

        scatterplot=sns.regplot(data=df_test,x='Rating_difference',y='XG_difference', scatter=False, ax=ax)
    fig = scatterplot.get_figure() 
    st.pyplot(fig)   
if page == pages[2]:
    st.markdown('# Modeling 1 : Simulate a Game')
    teams =['Qatar','Netherlands' ,'Senegal','Ecuador',
              'England','United States','Iran','Wales',
              'Argentina','Poland','Mexico','Saudi Arabia',
              'France','Denmark','Tunisia','Australia',
              'Spain','Japan','Germany','Costa Rica',
              'Belgium','Croatia','Morocco','Canada',
              'Brazil','Switzerland','Serbia','Cameroon',
              'Portugal','Uruguay','South Korea','Ghana']

    team_H = st.selectbox("Choose the Home Team",options=teams)
    team_A = st.selectbox("Choose the Away Team",options=[t for t in teams if t!=team_H])

    Team_Home = Team(team_H , df_test.loc[df_test['Team_A']==team_H,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_H,'Rating_A'].mode()[0])
    Team_Away = Team(team_A, df_test.loc[df_test['Team_A']==team_A,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_A,'Rating_A'].mode()[0])

    m = Match( Team_Home,Team_Away,df_test,False )
    st.write("The predicted results by the model are:\n", m)
    st.write("The predicted winner of this Match by the model is:", m.winner.name)
    
    fig_game =plt.figure()
    sns.barplot(x=[team_H,'Draw',team_A] ,y=m.results)
    plt.title("Predicted results of the game")
    plt.ylabel("Probability")
    st.pyplot(fig_game )
if page == pages[3]:
    st.markdown('# Modeling 2 : Simulate a Group Stage')
    teams =['Qatar','Netherlands' ,'Senegal','Ecuador',
              'England','United States','Iran','Wales',
              'Argentina','Poland','Mexico','Saudi Arabia',
              'France','Denmark','Tunisia','Australia',
              'Spain','Japan','Germany','Costa Rica',
              'Belgium','Croatia','Morocco','Canada',
              'Brazil','Switzerland','Serbia','Cameroon',
              'Portugal','Uruguay','South Korea','Ghana']
    #Selecting 4 teams for the group
    team_1 = st.selectbox("Choose the Home Team",options=teams)
    team_2 = st.selectbox("Choose the Away Team",options=[t for t in teams if t!=team_1])
    team_3 = st.selectbox("Choose the Away Team",options=[t for t in teams if t not in [team_1,team_2]])
    team_4 = st.selectbox("Choose the Away Team",options=[t for t in teams if t not in [team_1,team_2,team_3]])
    #Create the 4 teams
    Team_1 = Team(team_1 , df_test.loc[df_test['Team_A']==team_1 ,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_1 ,'Rating_A'].mode()[0])
    Team_2 = Team(team_2, df_test.loc[df_test['Team_A']==team_2,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_2,'Rating_A'].mode()[0])
    Team_3 = Team(team_3 , df_test.loc[df_test['Team_A']==team_3,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_3,'Rating_A'].mode()[0])
    Team_4 = Team(team_4, df_test.loc[df_test['Team_A']==team_4,'score_A'].mode()[0],df_test.loc[df_test['Team_A']==team_4,'Rating_A'].mode()[0])



    Group_A =Group_Stage([Team_1,Team_2,Team_3,Team_4],df_test)
    st.write("The qualified teams are",Group_A.first_qualified.name,'and','\n', Group_A.second_qualified.name)
    #st.write(Group_A.teams)
    #st.write([t.name for t in Group_A.teams] )
    #st.write([t.results["Points"] for t in Group_A.teams] )
    fig =plt.figure()
    sns.barplot(y= [t.name for t in Group_A.teams]  ,x=[t.results["Points"] for t in Group_A.teams],orient ="h")
    plt.title("Predicted results of the group stage")
    plt.ylabel("Points")
    st.pyplot(fig)

    fig2 =plt.figure()
    sns.barplot(x= [t.name for t in Group_A.teams]  ,y=[t.results["Goals"] - t.results["Goals_Against"] for t in Group_A.teams])
    plt.title("Predicted goal difference of the group stage")
    plt.ylabel("Goals - Goals Agaist")
    st.pyplot(fig2)

    fig3 =plt.figure()
    sns.barplot(x= [t.name for t in Group_A.teams]  ,y=[t.results["Goals"] for t in Group_A.teams])
    plt.title("Predicted number of goals for this group stage")
    plt.ylabel("# of Goals")
    st.pyplot(fig3)

if page == pages[4]:
    st.markdown('# Modeling 3 : Simulate the Tournament')
    st.header("The final results of the model based on Monte Carlo Simulation (for N = 50 simulations) : ") 
    #st.image('ouput.png')
    st.write("There is still improvements to be made... don' you think ;) ?")
    



   


