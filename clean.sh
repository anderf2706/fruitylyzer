# kjør ./clean.sh /path/to/file.py
# du må laste ned disse med pip:
# pycodestyle==2.5.0
# autopep8==1.5.0
# pylint

echo Sortering av .gitignore of requirements.txt er slått av midlertidig

ARG1="$1"
LENGTH=${#ARG1}


if [ $LENGTH -eq 0 ] 
then 
    echo "Spesifiser hvilken pakke du vil rydde opp i:"
    echo "./clean <pakkenavn>"
    echo "eller"
    echo "./clean <sti/til/fil.py>"
    exit 1
    
fi

if [[ $ARG1 == *"/"* ]]
then
    POSTFIX=""
    CLEAN_PATH=$ARG1
    NAME=$(echo $ARG1 | cut -d '/' -f 3 | cut -d '.' -f 1)

else
    POSTFIX="/*.py"
    CLEAN_PATH="packages/$ARG1"
    NAME=$ARG1

fi


echo Kjører autopep
autopep8 -i -r -a -a --exclude env ${CLEAN_PATH}  

echo Kjører PyLint
pylint --const-rgx='[a-z_][a-z0-9_]{2,30}$' --ignore=env $CLEAN_PATH${POSTFIX} > ${NAME}_pylint.log
echo Skrev resultater til ${NAME}_pylint.log
