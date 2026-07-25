cd submodules/cmft-master
CMFT=/path/to/HRAvatar/submodules/cmft-master/_build/linux64_gcc/bin/cmftRelease
envmap_path=/path/to/code/HRAvatar/assets/envmaps/autumn_field_puresky
eval $CMFT $@ --input "$envmap_path/autumn_field_puresky_1k.hdr"  \
       ::Filter options                  \
       --filter radiance                 \
       --srcFaceSize 256                 \
       --excludeBase false               \
       --mipCount 7                      \
       --generateMipChain false           \
       --glossScale 10                   \
       --glossBias 3                     \
       --lightingModel blinnbrdf         \
       --edgeFixup none                  \
       --dstFaceSize 256                 \
       ::Processing devices              \
       --numCpuProcessingThreads 8       \
       --useOpenCL true                  \
       --clVendor anyGpuVendor           \
       --deviceType gpu                  \
       --deviceIndex 0                   \
       ::Aditional operations            \
       --inputGammaNumerator 1.0         \
       --inputGammaDenominator 1.0       \
       --outputGammaNumerator 1.0        \
       --outputGammaDenominator 1.0      \
       ::Output                          \
        --outputNum 1                    \
        --output0 "$envmap_path/specular" \
        --output0params tga,bgr8,latlong 
 
eval $CMFT $@ --input "$envmap_path/autumn_field_puresky_1k.hdr"  \
       ::Filter options                  \
       --filter irradiance                 \
       --srcFaceSize 256                 \
       --dstFaceSize 128               \
       ::Aditional operations            \
       --inputGammaNumerator 1.0         \
       --inputGammaDenominator 1.0       \
       --outputGammaNumerator 1.0        \
       --outputGammaDenominator 1.0      \
       ::Output                          \
        --outputNum 1                    \
        --output0 "$envmap_path/diffuse" \
        --output0params tga,bgr8,latlong \