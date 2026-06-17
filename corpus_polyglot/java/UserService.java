// Demo Java class: a user service with CRUD-style methods.
// Часть полиглот-корпуса CodeLens (демонстрация поиска по Java через tree-sitter).

package com.codelens.demo;

import java.util.List;
import java.util.Optional;

/**
 * Application service that manages users and their credentials.
 */
public class UserService {

    private final UserRepository repository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository repository, PasswordEncoder passwordEncoder) {
        this.repository = repository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Register a new user, hashing the password before saving.
     */
    public User createUser(String email, String rawPassword) {
        String passwordHash = passwordEncoder.encode(rawPassword);
        User user = new User(email, passwordHash, true);
        return repository.save(user);
    }

    /**
     * Find a user by id or throw a NotFoundException.
     */
    public User getUserById(long userId) {
        Optional<User> found = repository.findById(userId);
        return found.orElseThrow(() -> new NotFoundException("User " + userId + " not found"));
    }

    /**
     * List active users with simple pagination.
     */
    public List<User> listActiveUsers(int skip, int limit) {
        return repository.findActive(skip, limit);
    }

    /**
     * Check whether the supplied password matches the stored hash.
     */
    public boolean checkPassword(User user, String rawPassword) {
        return passwordEncoder.matches(rawPassword, user.getPasswordHash());
    }
}
