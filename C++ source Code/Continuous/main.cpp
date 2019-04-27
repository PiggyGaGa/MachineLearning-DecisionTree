#include <iostream>
#include "CDTree.hpp"
#include "RFCSV.hpp"
#include <Eigen/Dense>
using Eigen::MatrixXf; using Eigen::MatrixXi;
int main()
{

	std::pair<MatrixXf, MatrixXi> trainData, testData;
	RFCSV<MatrixXf, MatrixXi> readcsv;
	trainData = readcsv.getData("../data/ContinuousTrain.csv");
	testData = readcsv.getData("../data/ContinuousTest.csv");
	CDTree myCDTree(4, 3, 0.0001);
	myCDTree.buildTree(trainData.first, trainData.second, "ID3");
	MatrixXi predictresult = myCDTree.predict(testData.first);
	float accuracy = 0;
	for (int i = 0; i < predictresult.size(); i++)
	{
		if (testData.second(i) == predictresult(i))
		{
			accuracy += 1.0 / predictresult.size();
		}
	}
	cout << "准确率为：" << accuracy * 100 << "\% 你很棒棒哦！" << endl;
}
