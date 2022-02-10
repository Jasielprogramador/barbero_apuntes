package github;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class crossValidationOneR {

	static long startTime = System.nanoTime();
	
	public static void main(String[] args) throws Exception{
		
		DataSource source = new DataSource(args[0]);
		Instances dataInstances =source.getDataSet();
		
		dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
		
		OneR modeOneR = new OneR();
		Evaluation evaluation = new Evaluation(dataInstances);
		evaluation.crossValidateModel(modeOneR, dataInstances, 5, new Random(1));
		
		System.out.println(dataInstances.numAttributes());
		System.out.println(dataInstances.numInstances());
		System.out.println(dataInstances.attributeStats(0).missingCount);
		System.out.println(dataInstances.numDistinctValues(0));
		
		File file = new File(args[1]);
		try {
			file.createNewFile();
			long endTime = System.nanoTime();
			FileWriter writer = new FileWriter(file);
			writer.write((endTime-startTime)/1000+" milisegundu \n");
			writer.write(args[1]+"\n");
			writer.write(evaluation.toMatrixString()+"\n");
			writer.write(evaluation.pctCorrect()+"\n");
			writer.write(evaluation.weightedFMeasure()+"\n");
			writer.write(evaluation.toSummaryString());
			writer.flush();
			writer.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
