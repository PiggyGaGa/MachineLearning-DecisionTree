/*
 read data from csv 
*/
#ifndef RFCSV_H
#define RFCSV_H
#include <iostream>
#include <string>
#include <vector>
using std::cout; using std::cin; using std::cerr; using std::endl;
using std::string; using std::ifstream; using std::istringstream;
using std::vector; using std::ios;

template<typename dataT, typename labelT>
class RFCSV
{
public:
    string csvFileName;
    RFCSV()
    {
        this->csvFileName = "";
    }
    RFCSV(string fileName)
    {
        this->csvFileName = fileName;
    }

    std::pair<dataT, labelT> getData(string fileName)
    {
        
        string csvFileName = fileName;

        int cols = 0, rows = 0;  //通过文件读取 获取行和列的信息。
        vector<vector<string>> strResult;
        ifstream read;
        read.open(csvFileName, ios::in);

        string headline;
        getline(read, headline);
        rows++;  //每次readline都要加行数一次
        if (rows > 0)
        {
            string attributeName;
            char delim = ',';
            istringstream stringin(headline);
            vector<string> oneLineString;
            while (getline(stringin, attributeName, delim))
            {
                cols++;
                oneLineString.push_back(attributeName);
            }
            strResult.push_back(oneLineString);
            //cout << endl;
        }
        //获取第一行信息
        string line;   //从第二行以后开始存储的每一行的数据
                        //下面获取后面的信息		
        while (getline(read, line))
        {
            rows++;
            int realcols = 0;  //用来判断此行中数据是否和标题行的数据列数相同，相同才存储，否则不存储
            string attributeName;
            char delim = ',';
            istringstream stringin(line);
            vector<string> oneLineString;
            while (getline(stringin, attributeName, delim))
            {
                realcols++;
                //cout << attributeName << "  ";
                oneLineString.push_back(attributeName);
            }
            if (realcols == cols)
            { //只有和标题行数据列数相同才存储
                strResult.push_back(oneLineString);
            }
        }
        dataT Data(rows, cols - 1);
        labelT label(rows, 1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols - 1; j++)
            {
                Data(i, j) = stringToNum<float>(strResult[i][j]);
            }
            label(i, 0) = stringToNum<int>(strResult[i][cols - 1]);
        }
        return std::pair<dataT, labelT>(Data, label);
        }

    std::pair<dataT, labelT> getData()
    {
        std::pair<dataT, labelT> result;
        string csvFileName = this->csvFileName;
        return result;
    }

    template <class Type>
    Type stringToNum(const string& str)
    {
        istringstream iss(str);
        Type num;
        iss >> num;
        return num;    
    }
 
};

#endif