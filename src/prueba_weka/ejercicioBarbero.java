package github;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ejercicioBarbero {
	
	/* ENUNCIADO:
	 * 	- Clasificador One R
	 	-Evaluacion 5 veces hold out
		-Que imprima matrixstring
		-La media de los 5 accuracy
		-La desviacion tipica de los 5 accuracy
	 */
	
	static long startTime = System.nanoTime();
	
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances dataInstances =source.getDataSet();
		
		dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
		
		double media = 0.0;
		double[] acumuladorDesv = new double[5];
		
		for(int i = 0;i<5;i++) {
		
			Randomize filteRandomize = new Randomize();
			filteRandomize.setInputFormat(dataInstances);
			Instances randomDataInstances= Filter.useFilter(dataInstances, filteRandomize);
			
			RemovePercentage filteRemovePercentage = new RemovePercentage();
			filteRemovePercentage.setInputFormat(randomDataInstances);
			filteRemovePercentage.setPercentage(30);
			Instances traInstances = Filter.useFilter(randomDataInstances, filteRemovePercentage);
			
			filteRemovePercentage = new RemovePercentage();
			filteRemovePercentage.setInputFormat(randomDataInstances);
			filteRemovePercentage.setPercentage(30);
			filteRemovePercentage.setInvertSelection(true);
			Instances test = Filter.useFilter(randomDataInstances, filteRemovePercentage);
			
			OneR modeOneR = new OneR();
			modeOneR.buildClassifier(traInstances);
			Evaluation evaluation = new Evaluation(traInstances);
			evaluation.evaluateModel(modeOneR, test);
			
			System.out.println(evaluation.toMatrixString());
			
			media = media + evaluation.pctCorrect();
			
			acumuladorDesv[i]=evaluation.pctCorrect();
			
		
		}
		
		media = media / 5;
		
		double desv = 0.0;
		
		for(int i = 0;i<5;i++) {
			desv = desv + (acumuladorDesv[i]-media)+(acumuladorDesv[i]-media);
		}
		desv = Math.sqrt(desv/5);
		
		
	}

}
