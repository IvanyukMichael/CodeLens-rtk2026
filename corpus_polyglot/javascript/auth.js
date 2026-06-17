// Demo JavaScript module: authentication helpers (Express + JWT style).
// Часть полиглот-корпуса CodeLens для демонстрации поиска по второму языку.

const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");

const ACCESS_TOKEN_TTL_MINUTES = 30;

/**
 * Create a signed JWT access token for the given user id.
 * The token expires after ACCESS_TOKEN_TTL_MINUTES minutes.
 */
function createAccessToken(userId, secret) {
  const payload = { sub: userId };
  return jwt.sign(payload, secret, {
    expiresIn: `${ACCESS_TOKEN_TTL_MINUTES}m`,
  });
}

/**
 * Hash a plaintext password using bcrypt with a generated salt.
 */
async function hashPassword(plainPassword) {
  const salt = await bcrypt.genSalt(12);
  return bcrypt.hash(plainPassword, salt);
}

/**
 * Verify a plaintext password against a stored bcrypt hash.
 */
async function verifyPassword(plainPassword, passwordHash) {
  return bcrypt.compare(plainPassword, passwordHash);
}

/**
 * Express middleware that rejects requests without a valid bearer token.
 */
const requireAuth = (secret) => (req, res, next) => {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  if (!token) {
    return res.status(401).json({ detail: "Not authenticated" });
  }
  try {
    req.user = jwt.verify(token, secret);
    return next();
  } catch (err) {
    return res.status(401).json({ detail: "Invalid token" });
  }
};

module.exports = { createAccessToken, hashPassword, verifyPassword, requireAuth };
