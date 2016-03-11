#ifndef __MAP_2_FILE_MAPPING_H__
#define __MAP_2_FILE_MAPPING_H__

#include "Names.h"
#include <string>
#include <iostream>
#include <vector>
#include <random>

class Map2FileMapping
{

public: 

	struct SubFieldRect
	{
		// bottom left corner
		int _rangeStart;
		int _rangeEnd;
		int _rowStart;
		int _rowEnd;
		int _ranges;
		int _rows;
		int _sum;

		SubFieldRect(int rangeStart, int rangeEnd, int rowStart, int rowEnd);

		void setParam(int rangeStart, int rangeEnd, int rowStart, int rowEnd);
	};

	struct FieldConfig
	{
		int name;
		int rowStart;
		int rowEnd;
		int rangeStart;
		int rangeEnd;
		int id;

		// rep 1 or rep 2
		std::vector<std::vector<std::vector<SubFieldRect>>> repVec;

		// short, tall, PS
		std::vector<std::vector<SubFieldRect>> typeVec;

		// 3 types
		std::vector<SubFieldRect> SFRVec_short;
		std::vector<SubFieldRect> SFRVec_tall;
		std::vector<SubFieldRect> SFRVec_ps;

		FieldConfig();

		bool addTypeSubFieldRect(int rep, int type, SubFieldRect sfr);
	};

	std::default_random_engine generator;
	
	std::vector<FieldConfig> fieldConfigVec;
		
	FieldConfig AHConfig;
	FieldConfig CHConfigPt1;
	FieldConfig CHConfigPt2;
	FieldConfig ASConfig;
	FieldConfig CSConfig;


	Map2FileMapping();

	void setSideRow(int row, int* fileRow, int* side);

	bool getFileName(int field, int range, int row, std::vector<std::string>& pathVec, int& plantSide);

	// number: number of images needed, plantLocation (range, row)
	bool getRandomImages(int field, int rep, int type, int number, std::vector<std::vector<int>>& plantLocationVec);

	bool checkRangeRowInBoundForType(int field, int type, int range, int row);

};


#endif