#######################################################################################
# Copyright: (C) 2021 Robotics Brain and Cognitive Sciences
# Author:  Giulia Belgiovine Gonzalez Jonas
# email:  giulia.belgiovine@iit.it
# Permission is granted to copy, distribute, and/or modify this program
# under the terms of the GNU General Public License, version 2 or any
# later version published by the Free Software Foundation.
#  *
# A copy of the license can be found at
# http://www.robotcub.org/icub/license/gpl.txt
#  *
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details
#######################################################################################

# This section sources the icub_basics.sh from icub interaction demos 
DEMOS_BASICS=$(yarp resource --context icubDemos --find icub_basics.sh | grep -v 'DEBUG' | tr -d '"')
echo sourcing $DEMOS_BASICS
source $DEMOS_BASICS


#######################################################################################
# "MAIN" DEMOS:                                                                    #
#######################################################################################
POSE_TIMING=10
CORRECTION_TIMING=4


go_home() {
    echo "set all hap" | yarp rpc /icub/face/emotions
    go_home_helper 3.0
    sleep 1.5

}

go_home_human(){

    echo "set all hap" | yarp rpc /icub/face/emotions/in
    breathers "stop"

    echo "ctpq time 2.0 off 0 pos (-1.4 15.8 16.0 15.0 -19.8 -0.32 -9.1 40.0 29.0 8.0 30.0 25.0 30.0 25.0 30.0 80.0)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 2.0 off 0 pos (-4.4 13.9 15.02 22.7 -6.7 -8.8 1.4 40.0 29.0 8.0 30.0 25.0 30.0 25.0 30.0 80.0)" | yarp rpc /ctpservice/right_arm/rpc
    echo "ctpq time 2.0 off 0 pos (-3.0 0.0 -8.0)" | yarp rpc /ctpservice/torso/rpc

    sleep 1.0
    echo "abs 0 -5 0" | yarp write ... /iKinGazeCtrl/angles:i
    breathers "start"

}

go_home_body() {

    echo "set all hap" | yarp rpc /icub/face/emotions
    go_home_body_helper 3.0
    sleep 1.5
}

go_home_body_helper() {
    # This is with the arms close to the legs
    # echo "ctpq time $1 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/right_arm/rpc
    # echo "ctpq time $1 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/left_arm/rpc
    # This is with the arms over the table
    go_home_helperR $1
    go_home_helperL $1
    go_home_helperT $1
    go_home_helperLL $1
    go_home_helperRL $1
}

go_home_helper() {
    # This is with the arms close to the legs
    # echo "ctpq time $1 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/right_arm/rpc
    # echo "ctpq time $1 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/left_arm/rpc
    # This is with the arms over the table
    go_home_helperR $1
    go_home_helperL $1
    go_home_helperH $1
    go_home_helperT $1
    go_home_helperLL $1
    go_home_helperRL $1
}

go_home_helperL(){
    # echo "ctpq time $1 off 0 pos (-30.0 36.0 0.0 60.0 0.0 0.0 0.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time $1 off 0 pos (-6.0 23.0 4.0 63.0 -24.0 -3.0 -3.0 40.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/left_arm/rpc
}

go_home_helperR(){
    # echo "ctpq time $1 off 0 pos (-30.0 36.0 0.0 60.0 0.0 0.0 0.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/right_arm/rpc
    echo "ctpq time $1 off 0 pos (-6.0 23.0 4.0 63.0 -24.0 -3.0 -3.0 40.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/right_arm/rpc
}

go_home_helperH(){
        echo "abs 0 -8 0" | yarp write ... /iKinGazeCtrl/angles:i
}

go_home_helperT(){
    echo "ctpq time $1 off 0 pos (-3.0 0.0 -8.0)" | yarp rpc /ctpservice/torso/rpc
}

go_home_helperLL(){
        echo "ctpq time $1 off 0 pos (0.0 0.0 0.0 0.0 -0.17578167915449 0.258179341258157) " | yarp rpc /ctpservice/left_leg/rpc
}

go_home_helperRL(){
	echo "ctpq time $1 off 0 pos (0.0 0.0 0.0 0.0 -0.17578167915449 0.258179341258157) " | yarp rpc /ctpservice/right_leg/rpc
}

move_head(){
    echo "ctpq time 2.0 off 0 pos (-12.0 0.0 0.0)" | yarp rpc /ctpservice/head/rpc
}

pointRightPose(){

	echo "ctpq time 2.0 off 0 pos (-22.28 46.99 -10.78 24.02 -59.60 7.50 -13.88 37.09 10.39 55.14 0.0 0.0 0.37 0.0 0.0 2.34)" | yarp rpc /ctpservice/right_arm/rpc
	echo "ctpq time 2.0 off 0 pos (15 0 0)" | yarp rpc /ctpservice/torso/rpc
    sleep 1.5
    go_home_helperR 2
    go_home_helperT 2
}

pointCenterPose(){

	echo "ctpq time 2.0 off 0 pos (-65.83 18.12 -7.82 14.98 -54.01 -0.37 1.04 41.40 60.48 56.0 1.87 0.0 0.0 0.0 0.37 0.46)" | yarp rpc /ctpservice/right_arm/rpc
    sleep 1.5
    go_home_helperR 2
}

pointLeftPose(){

	echo "ctpq time 2.0 off 0 pos (-12.8 36.3 -15.8 39.2 -54.8 6.3 -12.8 34.0 10.3 25.7 0.37 30.48 50.2 42.6 2.9 27.37)" | yarp rpc /ctpservice/left_arm/rpc
	echo "ctpq time 2.0 off 0 pos (-13 0 0)" | yarp rpc /ctpservice/torso/rpc
    sleep 1.5
    go_home_helperL 2
    go_home_helperT 2
}

saluta() {

    echo "ctpq time 1.5 off 0 pos (-60.0 44.0 -2.0 96.0 53.0 -17.0 -11.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc
    sleep 1.0
    # echo "ctpq time 0.3 off 0 pos (-60.0 44.0 -2.0 96.0 53.0 -17.0  25.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc
    # sleep 0.5
    # echo "ctpq time 0.3 off 0 pos (-60.0 44.0 -2.0 96.0 53.0 -17.0 -11.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc
    # sleep 0.5
    # echo "ctpq time 0.3 off 0 pos (-60.0 44.0 -2.0 96.0 53.0 -17.0  25.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc
    # sleep 0.5
    # echo "ctpq time 0.3 off 0 pos (-60.0 44.0 -2.0 96.0 53.0 -17.0 -11.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc

    # sleep 1.0

    go_home
}

happy() {
    echo "set all hap" | yarp rpc /icub/face/emotions/in
}

surprised() {
    echo "set mou sur" | yarp rpc /icub/face/emotions/in
    echo "set leb sur" | yarp rpc /icub/face/emotions/in
    echo "set reb sur" | yarp rpc /icub/face/emotions/in
}

neutral() {
    echo "set mou neu" | yarp rpc /icub/face/emotions/in
    echo "set leb neu" | yarp rpc /icub/face/emotions/in
    echo "set reb neu" | yarp rpc /icub/face/emotions/in
}

sad() {
    echo "set mou sad" | yarp rpc /icub/face/emotions/in
    echo "set leb sad" | yarp rpc /icub/face/emotions/in
    echo "set reb sad" | yarp rpc /icub/face/emotions/in
}

cun() {
    echo "set mou neu" | yarp rpc /icub/face/emotions/in
    echo "set reb cun" | yarp rpc /icub/face/emotions/in
    echo "set leb cun" | yarp rpc /icub/face/emotions/in
}

angry() {
    echo "set all ang" | yarp rpc /icub/face/emotions/in
}

############ YOGA POSES ##############

guerriero_pose(){
    echo "ctpq time 3.5 off 0 pos (6.59181296829338 42.6819889696996 70.1368899826415 -58.8868625167542 -3.95508778097603 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5 off 0 pos (-0.0878908395772451 37.7930610182154 0.0 -0.0878908395772451 -0.17578167915449 -20.3961679593944) " | yarp rpc /ctpservice/right_leg/rpc

    
    echo "ctpq time 3.5 off 0 pos (7.47072136406583 75.426819889697 -6.50392212871613 14.8525850893191 10.4205576673771 -5.55909560326075 12.6507877216497 19.3524642394146 29.822460504054 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (7.0312671661796 81.4308628683175 -10.5963393465316 15.0293335677089 9.0637428314034 10.4809826195865 1.05469007492694 27.0923512996858 27.5867372723078 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  
    
    sleep $POSE_TIMING

    go_home
    
}

lottatore_pose(){
    echo "ctpq time 3.5 off 0 pos (6.59181296829338 42.6819889696996 70.1368899826415 -58.8868625167542 -3.95508778097603 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5 off 0 pos (-0.0878908395772451 37.7930610182154 0.0 -0.0878908395772451 -0.17578167915449 -20.3961679593944) " | yarp rpc /ctpservice/right_leg/rpc

    
    echo "ctpq time 3.5 off 0 pos (7.47072136406583 75.426819889697 -6.50392212871613 14.8525850893191 10.4205576673771 -5.55909560326075 12.6507877216497 19.3524642394146 29.822460504054 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (7.0312671661796 81.4308628683175 -10.5963393465316 15.0293335677089 9.0637428314034 10.4809826195865 1.05469007492694 27.0923512996858 27.5867372723078 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  


    echo "ctpq time 3.5 off 0 pos (-8.9 30.7617938520358 0.00549317747357782)" | yarp rpc /ctpservice/torso/rpc

    sleep $POSE_TIMING

    go_home

}

triangolo_pose(){
    echo "ctpq time 3.5 off 0 pos (6.67970380787062 34.3323592098614 70.1368899826415 -6.06446793082991 30.3223396541495 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5 off 0 pos (-0.17578167915449 40.8692404034189 -0.0878908395772451 -0.35156335830898 -0.17578167915449 -20.1324954406627) " | yarp rpc /ctpservice/right_leg/rpc

    
    echo "ctpq time 3.5 off 0 pos (7.11915800575685 59.4306870866384 12.6562808991233 14.9853881479203 10.4205576673771 -6.61378567818769 9.04726329898266 18.825 29.9817 38.452 0.0 0.0 1.186 0.0 0.0 0.0 )" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (-2.54883434774011 105.425062072905 16.3861484036826 14.9414427281317 -3.99903320076465 10.3930917800092 0.878908395772451 27.2077080266309 27.8174507261981 -0.444947375359803 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  


    echo "ctpq time 3.5 off 0 pos (0.35156335830898 31.552811408231 0.00549317747357782)" | yarp rpc /ctpservice/torso/rpc

    sleep $POSE_TIMING

    go_home

}

granchio_pose(){
    echo "ctpq time 3.5 off 0 pos (13.0078442574323 59.1175759706445 70.0489991430643 -64.4239854101206 0.439454197886225 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5 off 0 pos (23.0273999692382 56.5138098481686 67.1486014370152 -64.0724220518116 -0.17578167915449 6.84999230955154) " | yarp rpc /ctpservice/right_leg/rpc

    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $POSE_TIMING

    go_home
}

albero_pose(){
    
    echo "ctpq time 3.5  off 0 pos (14.1504251719365 59.2054668102217 70.0489991430643 -98.613522005669 29.7071037771088 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $POSE_TIMING

    go_home
}

cactus_pose(){
    
    echo "ctpq time 3.5  off 0 pos (17.22660455714 67.3793148909055 65.8302388433566 -2.28516182900837 17.7539495946035 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    
    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $POSE_TIMING

    go_home

}


aeroplano_pose(){
    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc  


    echo "ctpq time 3.5 off 0 pos (-0.263672518731735 -1.23047175408143 59.7712640900002)" | yarp rpc /ctpservice/torso/rpc

    sleep $POSE_TIMING

    go_home
}


fenicottero_pose(){
    echo "ctpq time 3.5 off 0 pos (-1.58203511239041 0.0 0.0 -98.9650853639779 -0.17578167915449 -0.0933840170508229) " | yarp rpc /ctpservice/right_leg/rpc


    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc  


    echo "ctpq time 3.5 off 0 pos (-0.263672518731735 0.0 0.0)" | yarp rpc /ctpservice/torso/rpc

    sleep $POSE_TIMING

    go_home
}

ruota_pose(){

    echo "ctpq time 3.5 off 0 pos (0.0 -0.0329590648414669 0.0 0.0 0.263672518731735 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc
    echo "ctpq time 3.5 off 0 pos (1.66992595196766 77.2579815868691 0.0 -2.19727098943113 -0.17578167915449 -0.0933840170508229) " | yarp rpc /ctpservice/right_leg/rpc

    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc  

 
    echo "ctpq time 3.5 off 0 pos (0.439454197886225 30.6739030124585 1.76330996901848)" | yarp rpc /ctpservice/torso/rpc

    sleep $POSE_TIMING

    go_home
}

##### CORRECTIONS #####

# Corrections related to the AEROPLANE pose: 

aeroplano_torso(){

    # This is with the arms close to the legs
    echo "ctpq time 3.0 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/right_arm/rpc
    echo "ctpq time 3.0 off 0 pos (-6.0 23.0 25.0 29.0 -24.0 -3.0 -3.0 19.0 29.0 8.0 30.0 32.0 42.0 50.0 50.0 114.0)" | yarp rpc /ctpservice/left_arm/rpc
    echo "ctpq time 3.5 off 0 pos (-0.263672518731735 -1.23047175408143 59.7712640900002)" | yarp rpc /ctpservice/torso/rpc
    
    sleep $CORRECTION_TIMING

    go_home

}

aeroplano_armL(){

    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc  
    
    sleep $CORRECTION_TIMING

    go_home
}

aeroplano_armR(){

    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc  
    
    sleep $CORRECTION_TIMING

    go_home
}

# Corrections related to the CRAB pose: 

granchio_armL(){

    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $CORRECTION_TIMING

    go_home
}

granchio_armR(){

    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc 

    sleep $CORRECTION_TIMING

    go_home
}

granchio_legL(){

    echo "ctpq time 3.5 off 0 pos (23.0273999692382 56.5138098481686 67.1486014370152 -64.0724220518116 -0.17578167915449 6.84999230955154) " | yarp rpc /ctpservice/right_leg/rpc


    sleep $CORRECTION_TIMING

    go_home
}

granchio_legR(){

    echo "ctpq time 3.5 off 0 pos (13.0078442574323 59.1175759706445 70.0489991430643 -64.4239854101206 0.439454197886225 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc


    sleep $CORRECTION_TIMING

    go_home
}

# Corrections related to the WARRIOR pose: 

guerriero_armL(){

    echo "ctpq time 3.5 off 0 pos (7.0312671661796 81.4308628683175 -10.5963393465316 15.0293335677089 9.0637428314034 10.4809826195865 1.05469007492694 27.0923512996858 27.5867372723078 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  
    
    sleep $CORRECTION_TIMING

    go_home
    
}

guerriero_armR(){

    echo "ctpq time 3.5 off 0 pos (7.47072136406583 75.426819889697 -6.50392212871613 14.8525850893191 10.4205576673771 -5.55909560326075 12.6507877216497 19.3524642394146 29.822460504054 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/left_arm/rpc
    
    sleep $CORRECTION_TIMING

    go_home
    
}

guerriero_legL(){

    echo "ctpq time 3.5 off 0 pos (-0.0878908395772451 37.7930610182154 0.0 -0.0878908395772451 -0.17578167915449 -20.3961679593944) " | yarp rpc /ctpservice/right_leg/rpc
  
    sleep $CORRECTION_TIMING

    go_home
    
}

guerriero_legR(){

    echo "ctpq time 3.5 off 0 pos (6.59181296829338 42.6819889696996 70.1368899826415 -58.8868625167542 -3.95508778097603 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

# Corrections related to the TREE pose: 

albero_armL(){

    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $CORRECTION_TIMING

    go_home
}

albero_armR(){
    
    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc

    sleep $CORRECTION_TIMING

    go_home
}

albero_legR(){
    
    echo "ctpq time 3.5  off 0 pos (14.1504251719365 59.2054668102217 70.0489991430643 -98.613522005669 29.7071037771088 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc  

    sleep $CORRECTION_TIMING

    go_home
}

# Corrections related to the FLAMINGO pose: 

fenicottero_torso(){

    echo "ctpq time 3.5 off 0 pos (-0.263672518731735 0.0 0.0)" | yarp rpc /ctpservice/torso/rpc

    sleep $CORRECTION_TIMING

    go_home

}

fenicottero_armL(){

    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $CORRECTION_TIMING

    go_home

}

fenicottero_armR(){

    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc

    sleep $CORRECTION_TIMING

    go_home

}

fenicottero_legL(){
    echo "ctpq time 3.5 off 0 pos (-1.58203511239041 0.0 0.0 -98.9650853639779 -0.17578167915449 -0.0933840170508229) " | yarp rpc /ctpservice/right_leg/rpc

    sleep $CORRECTION_TIMING

    go_home

}

# Corrections related to the FIGHTER pose: 

lottatore_torso(){

    echo "ctpq time 3.5 off 0 pos (-8.9 30.7617938520358 0.00549317747357782)" | yarp rpc /ctpservice/torso/rpc

    sleep $CORRECTION_TIMING

    go_home

}

lottatore_armL(){

    echo "ctpq time 3.5 off 0 pos (7.0312671661796 81.4308628683175 -10.5963393465316 15.0293335677089 9.0637428314034 10.4809826195865 1.05469007492694 27.0923512996858 27.5867372723078 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc  

    sleep $CORRECTION_TIMING

    go_home

}

lottatore_armR(){
   
    echo "ctpq time 3.5 off 0 pos (7.47072136406583 75.426819889697 -6.50392212871613 14.8525850893191 10.4205576673771 -5.55909560326075 12.6507877216497 19.3524642394146 29.822460504054 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/left_arm/rpc 

    sleep $CORRECTION_TIMING

    go_home

}

lottatore_legL(){

    echo "ctpq time 3.5 off 0 pos (-0.0878908395772451 37.7930610182154 0.0 -0.0878908395772451 -0.17578167915449 -20.3961679593944) " | yarp rpc /ctpservice/right_leg/rpc  

    sleep $CORRECTION_TIMING

    go_home

}

lottatore_legR(){

    echo "ctpq time 3.5 off 0 pos (6.59181296829338 42.6819889696996 70.1368899826415 -58.8868625167542 -3.95508778097603 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc


    sleep $CORRECTION_TIMING

    go_home

}

# Corrections related to the CACTUS pose:

cactus_legR(){

    echo "ctpq time 3.5  off 0 pos (17.22660455714 67.3793148909055 65.8302388433566 -2.28516182900837 17.7539495946035 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc

    sleep $CORRECTION_TIMING

    go_home

}

cactus_armR(){

    echo "ctpq time 3.5  off 0 pos (-94.2189800268067 91.1592801740239 14.3262068510909 94.6144888049043 -2.46643668563644 -6.78956735734218 8.78359078025093 22.5000549317747 28.2843708114522 0.818483443563095 0.0 0.0 0.791017556195206 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc

    sleep $CORRECTION_TIMING

    go_home

}

cactus_armL(){

    echo "ctpq time 3.5  off 0 pos (-94.5705433851157 91.5383094197007 14.2767682538287 94.5705433851157 0.00549317747357782 -0.417481487991914 -0.791017556195206 25.1148074091978 29.8828854562633 0.0 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc

    sleep $CORRECTION_TIMING

    go_home

}

# Corrections related to the WHEEL pose:

ruota_legL(){

    echo "ctpq time 3.5 off 0 pos (1.66992595196766 77.2579815868691 0.0 -2.19727098943113 -0.17578167915449 -0.0933840170508229) " | yarp rpc /ctpservice/right_leg/rpc

    sleep $CORRECTION_TIMING

    go_home

}

ruota_legR(){

    echo "ctpq time 3.5 off 0 pos (0.0 -0.0329590648414669 0.0 0.0 0.263672518731735 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

ruota_armL(){

    echo "ctpq time 3.5 off 0 pos (6.2402496099844 75.7179582957966 0.82946979851025 17.1442068950364 8.61330227857002 -5.95460438135835 2.54883434774011 26.1640043066511 22.9889477269232 0.878908395772451 0.0 0.0 0.373536068203291 0.411988310518336 0.0 1.42273296565665)" | yarp rpc /ctpservice/right_arm/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

ruota_armR(){

    echo "ctpq time 3.5 off 0 pos (4.04297862055327 77.2725275208191 -4.13086946013052 18.1494583727011 4.4824328184395 -3.62549713256136 10.5414075717958 16.4630528883127 30.3333260090967 0.0 0.0 0.0 1.58203511239041 0.0 0.0 0.466920085254114)" | yarp rpc /ctpservice/left_arm/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

ruota_torso(){

    echo "ctpq time 3.5 off 0 pos (0.439454197886225 30.6739030124585 1.76330996901848)" | yarp rpc /ctpservice/torso/rpc

    sleep $CORRECTION_TIMING

    go_home

}

# Corrections related to the TRIANGLE pose:

triangolo_legL(){

    echo "ctpq time 3.5 off 0 pos (-0.17578167915449 40.8692404034189 -0.0878908395772451 -0.35156335830898 -0.17578167915449 -20.1324954406627) " | yarp rpc /ctpservice/right_leg/rpc

    sleep $CORRECTION_TIMING

    go_home

}

triangolo_legR(){

    echo "ctpq time 3.5 off 0 pos (6.67970380787062 34.3323592098614 70.1368899826415 -6.06446793082991 30.3223396541495 0.296631583573202) " | yarp rpc /ctpservice/left_leg/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

triangolo_armL(){

    echo "ctpq time 3.5 off 0 pos (-2.54883434774011 105.425062072905 16.3861484036826 14.9414427281317 -3.99903320076465 10.3930917800092 0.878908395772451 27.2077080266309 27.8174507261981 -0.444947375359803 0.0 0.0 0.0 0.0 0.0 0.0)" | yarp rpc /ctpservice/right_arm/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

triangolo_armR(){

    echo "ctpq time 3.5 off 0 pos (7.11915800575685 59.4306870866384 12.6562808991233 14.9853881479203 10.4205576673771 -6.61378567818769 9.04726329898266 18.825 29.9817 38.452 0.0 0.0 1.186 0.0 0.0 0.0 )" | yarp rpc /ctpservice/left_arm/rpc

    sleep $CORRECTION_TIMING

    go_home
    
}

triangolo_torso(){

    echo "ctpq time 3.5 off 0 pos (0.35156335830898 31.552811408231 0.00549317747357782)" | yarp rpc /ctpservice/torso/rpc

    sleep $CORRECTION_TIMING

    go_home

}

# No movement during the correction of a pose:

rest_position(){

    sleep $CORRECTION_TIMING

}

bravissimo(){

    sleep $CORRECTION_TIMING

}


#######################################################################################
# "MAIN" FUNCTION:                                                                    #
#######################################################################################
list() {
	compgen -A function
}

echo "********************************************************************************"
echo ""

$1 "$2"

if [[ $# -eq 0 ]] ; then 
    echo "No options were passed!"
    echo ""
    usage
    exit 1
fi


