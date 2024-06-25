package com.yanhuo.common.constant;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;

/**
 * @author chen
 */
@ApiModel("用户常量")
public interface UserConstant {

    @ApiModelProperty("用户id")
    String USER_ID = "userId";
}
