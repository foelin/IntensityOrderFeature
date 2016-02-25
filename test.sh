#!/bin/zsh
imgFile="img1.ppm"
ptFile="img1.haraff"

echo "LIOP"
./IntensityOrderFeature -img  $imgFile  -region $ptFile -des $ptFile".liop"  -type liop -liopRegionNum 6 -lsRadius 6 -liopNum 4 -liopType 1


echo "OIOP"
./IntensityOrderFeature -img  $imgFile  -region $ptFile -des $ptFile".oiop"  -type oiop -lsRadius 6 -oiopNum 3 -oiopType 1 -oiopRegionNum 4 -oiopQuantLevel 4

echo "MIOP"
./IntensityOrderFeature -img  $imgFile  -region $ptFile -des $ptFile".miop" -type miop -liopNum 4 -liopRegionNum 6 -liopType 1 -oiopNum 3 -oiopRegionNum 4 -oiopQuantLevel 4 -oiopType 1 -isApplyPCA 1 -pcaBasisNum 128 -pcaFile pca_miop.txt

echo "MIOP_FAST"
./IntensityOrderFeature -img  img1.ppm  -region $ptFile -des $ptFile".miop_fast"  -type miop_fast -liopNum 4 -liopRegionNum 6 -liopType 1 -oiopNum 3 -oiopRegionNum 4 -oiopQuantLevel 4 -oiopType 1 -isApplyPCA 1 -pcaBasisNum 128 -pcaFile pca_miop.txt

