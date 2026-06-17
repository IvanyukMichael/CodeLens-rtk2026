// Demo Java class: a REST controller exposing login / token endpoints.
// Демонстрирует извлечение класса, конструктора и методов (Class.method).

package com.codelens.demo;

/**
 * REST controller that issues JWT access tokens on successful login.
 */
public class AuthController {

    private static final int ACCESS_TOKEN_TTL_MINUTES = 30;

    private final UserService userService;
    private final JwtService jwtService;

    public AuthController(UserService userService, JwtService jwtService) {
        this.userService = userService;
        this.jwtService = jwtService;
    }

    /**
     * Authenticate a user and return a signed access token.
     */
    public TokenResponse loginForAccessToken(String email, String password) {
        User user = userService.getUserByEmail(email);
        if (!userService.checkPassword(user, password)) {
            throw new UnauthorizedException("Incorrect email or password");
        }
        String token = jwtService.createAccessToken(user.getId(), ACCESS_TOKEN_TTL_MINUTES);
        return new TokenResponse(token, "bearer");
    }

    /**
     * Return the profile of the currently authenticated user.
     */
    public User readCurrentUser(String bearerToken) {
        long userId = jwtService.parseSubject(bearerToken);
        return userService.getUserById(userId);
    }
}
