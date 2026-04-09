package com.creditscoring.model;

/**
 * User roles for the credit scoring system (matches original 4-role RBAC).
 */
public enum Role {
    ADMIN,
    MODEL_BUILDER,
    VALIDATOR,
    SCORER
}
