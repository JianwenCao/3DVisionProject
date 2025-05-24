mkdir -p data
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip

cd Replica
wget https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip
unzip cull_replica_mesh.zip
rm cull_replica_mesh.zip
mv cull_replica_mesh gt_mesh_culled

mkdir -p gt_mesh
mv *.ply gt_mesh
