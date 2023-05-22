# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

#!/usr/bin/env bash
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
# set region

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -eq 4 ]; then
    region=$1
    image=$2
    tag=$3
    DIR=$4
else
    echo "usage: $0 <aws-region> $1 <image-repo> $2 <image-tag>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

if [[ $region =~ ^cn.* ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:${tag}"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
fi

# Get the login command from ECR and execute it directly
if [[ $region =~ ^cn.* ]]
then
    $(aws ecr get-login --region ${region} --registry-ids 727897471807 --no-include-email)
else
    $(aws ecr get-login --region ${region} --registry-ids 763104351884 --no-include-email)
fi

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "creating ECR repository : ${fullname} "
    aws ecr create-repository --region ${region} --repository-name "${image}" > /dev/null
fi

$(aws ecr get-login --no-include-email --region ${region}  --registry-ids 763104351884)
docker build ${DIR}/ -t ${image} -f ${DIR}/Dockerfile  --build-arg region=${region}
docker tag ${image} ${fullname}

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)
docker push ${fullname}
if [ $? -eq 0 ]; then
	echo "Amazon ECR URI: ${fullname}"
else
	echo "Error: Image build and push failed"
	exit 1
fi
