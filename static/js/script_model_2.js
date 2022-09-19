function showLossFunctionEquation(){

    let select_loss = document.getElementById("select_loss");
    let MSE_equation = document.getElementById("MSE");
    let RMSE_equation = document.getElementById("RMSE");
    let CROSSENTROPYLOSS_equation_b = document.getElementById("CROSSENTROPYLOSS_b");
    let CROSSENTROPYLOSS_equation_m = document.getElementById("CROSSENTROPYLOSS_m");

    MSE_equation.style.display = "none";
    RMSE_equation.style.display = "none";
    CROSSENTROPYLOSS_equation_b.style.display = "none";
    CROSSENTROPYLOSS_equation_m.style.display = "none";

    if (select_loss.value == "MSE"){
      MSE_equation.style.display = "block";
    }else if (select_loss.value == "RMSE"){
      RMSE_equation.style.display = "block";
    }else if (select_loss.value == "CROSSENTROPYLOSS"){
      CROSSENTROPYLOSS_equation_b.style.display = "block";
      CROSSENTROPYLOSS_equation_m.style.display = "block";
    }
}

function showInitializingParameters(){

  let select_initializingRule = document.getElementById("select_initializingRule");
  let regressionParameters = document.getElementById("regressionParameters");

  if (select_initializingRule.value == "LinearRegression"){
    regressionParameters.style.display = "block";
  }else{
    regressionParameters.style.display = "none";
  }

}

function showReorganzingingParameters(){

    let reorganizingRule = document.getElementById("reorganizingRule");
    let reorganizingParamaters = document.getElementById("reorganizingParamaters");

    reorganizingParamaters.style.display = "none";

    if (reorganizingRule.value != "Disabled"){
      reorganizingParamaters.style.display = "block";
    }
  }


function showMatchingParameters(){

  let select_matchingRule = document.getElementById("select_matchingRule");
  let matchingLearningGoal = document.getElementById("p_matchingLearningGoal");
  let matchingLearningRateLowerBound = document.getElementById("p_matchingLearningRateLowerBound");
  

  if (select_matchingRule.value == "EU_LG"){
    matchingLearningGoal.style.display = "block";
    matchingLearningRateLowerBound.style.display = "none";
  }else if (select_matchingRule.value == "EU_LG_UA"){
    matchingLearningGoal.style.display = "block";
    matchingLearningRateLowerBound.style.display = "block";
  }
  else if (select_matchingRule.value == "Disabled"){
    matchingLearningGoal.style.display = "none";
    matchingLearningRateLowerBound.style.display = "none";
  }

}
