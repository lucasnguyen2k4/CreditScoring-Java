package com.creditscoring.config;

import com.creditscoring.model.Role;
import com.creditscoring.model.User;
import com.creditscoring.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

/**
 * Seeds default users on first startup (if MongoDB is empty).
 * Same 4 default users as the original Streamlit project.
 */
@Component
@RequiredArgsConstructor
public class DataSeeder implements CommandLineRunner {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) {
        if (userRepository.count() > 0) {
            System.out.println("✓ Database already has users, skipping seed.");
            return;
        }

        System.out.println("🌱 Seeding default users...");

        seedUser("admin", "admin123", "Admin User", Role.ADMIN);
        seedUser("builder", "builder123", "Model Builder", Role.MODEL_BUILDER);
        seedUser("validator", "validator123", "Validator User", Role.VALIDATOR);
        seedUser("scorer", "scorer123", "Scorer User", Role.SCORER);

        System.out.println("✓ Seeded 4 default users.");
    }

    private void seedUser(String username, String password, String displayName, Role role) {
        User user = new User();
        user.setUsername(username);
        user.setPasswordHash(passwordEncoder.encode(password));
        user.setDisplayName(displayName);
        user.setRole(role);
        user.setEnabled(true);
        userRepository.save(user);
        System.out.println("  → Created " + username + " (" + role + ")");
    }
}
