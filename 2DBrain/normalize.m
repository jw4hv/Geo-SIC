clear;
FilesV=dir('./*.mhd');
for k=1:size(FilesV)
    FileNames=FilesV(k).name;
    Filefolder=FilesV(k).folder;
    path=strcat( Filefolder,'/',FileNames) ;
    alldataV(k,:,:)=loadMETA(path);
    A = reshape(alldataV(k,:,:),128,128);
    normA = A - min(A(:));
    normA = normA ./ max(normA(:));
    writeMETA( normA,FileNames);
end