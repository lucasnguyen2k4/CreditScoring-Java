package com.creditscoring.dto;

import com.creditscoring.model.Role;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AuthResponse {
    private String token;
    private String username;
    private String displayName;
    private Role role;
}
